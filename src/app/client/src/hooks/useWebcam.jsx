import { createContext, useCallback, useEffect, useRef, useState } from "react";
import { io, Socket } from "socket.io-client";
import getFeatures from "../util/getFeatures";
import { Hands } from "@mediapipe/hands";
import { Camera } from '@mediapipe/camera_utils';

// var LIMIT_FPS = 30;

function useWebcam({
    numHands,
    dynamic,
    onCaptureError,
    handedness, // for left hand folks
    onResult
}) {
    const landmarkHistory = useRef([]);
    const [captureState, setCaptureState] = useState(false);
    const socket = io("http://127.0.0.1:5000");
    const hands = useRef(null);
    const webcamVideo = useRef(null);
    const mediapipeCamera = useRef(null),
    SocketContext = createContext<Socket>(socket);

    const date = new Date();

    const startTime = useRef(date.getTime() / 1000)
    const prevTime = useRef(0);
    const deltaTime = useRef(0);
    const frameCount = useRef(0);
    const totalFPS = useRef(0);

    const FPS_THROTTLE = useRef(30.0);

    const throttle = useCallback((callback) => {
        console.log("throttle", FPS_THROTTLE)
        let waiting = false;
        return function (...args) {
            if (!waiting) {
              callback.apply(this, args);
              waiting = true;
              setTimeout(function () {
                waiting = false;
              }, FPS_THROTTLE.current);
            }
        };
    }, [FPS_THROTTLE])

    const onResults = (results) => {
        if(prevTime.current === 0) {
            const newDate = new Date();
            prevTime.current = newDate.getTime() / 1000;
        }else {
            const newDate = new Date();
            const curTime = newDate.getTime() / 1000;
            const prevDeltaTime = curTime - prevTime.current;

            deltaTime.current = newDate.getTime() / 1000 - startTime.current;

            // const avgFPS = frameCount.current / deltaTime.current;
            const currentFPS = 1 / prevDeltaTime;
            totalFPS.current += currentFPS;
            const avgFPS = totalFPS.current / frameCount.current;

            frameCount.current += 1;
            prevTime.current = curTime;

            if(avgFPS > 14.5) {
                FPS_THROTTLE.current += 0.5;
            } else if(avgFPS < 13.5) {
                FPS_THROTTLE.current -= 0.5;
            }

            console.log("FRAMERATE:", currentFPS, "AVG FPS:", avgFPS, "THROTTLE:", FPS_THROTTLE.current);
        }
        const { multiHandLandmarks, multiHandedness } = results;
        if(multiHandLandmarks.length >= numHands) {
            let totalHandFeatures = [];
            multiHandLandmarks.forEach((landmarks) => {
                if(landmarks.length < numHands * 21) {
                    onCaptureError();
                    return;
                }
                const features = getFeatures(landmarks);
                totalHandFeatures = totalHandFeatures.concat(features);
            })
            if(dynamic) {
                if(captureState) {
                    console.log("RECORDING");
                    if(landmarkHistory.current.length < 30) {
                        console.log(totalHandFeatures);
                        landmarkHistory.current.push(totalHandFeatures);
                    }else {
                        console.log(landmarkHistory.current, landmarkHistory.current.length);
                        socket.emit('dynamic', { 
                            landmarkHistory: landmarkHistory.current,
                            reflect: handedness === "left",
                            numHands: numHands,
                        })
                        setCaptureState(false);
                    }
                }
            }else {
                socket.emit('stream', { 
                    features: totalHandFeatures,
                    reflect: handedness === "left",
                    numHands: numHands
                });
            }
        }
    }

    useEffect(() => {
        landmarkHistory.current = [];
        if(hands.current) hands.current.onResults(throttle(onResults));
        if(captureState && !dynamic) {
            throw new Error("useWebcam cannot start capture if webcam is not dynamic");
        }
    }, [captureState, FPS_THROTTLE])

    const loadHands = () => {
        try{
        if(!hands.current) {
            hands.current = new Hands({
                locateFile: (file) => {
                    return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`;
                }
            })
            hands.current.setOptions({
                maxNumHands: numHands,
                modelComplexity: 1,
                minDetectionConfidence: 0.5,
                minTrackingConfidence: 0.5,
            })
        }
        hands.current.onResults(throttle(onResults));
    }catch(error){

    }
    }

    const parseData = (data) => {
        const deserialized = JSON.parse(data);
        return deserialized;
    }

    const teardown = () => {
        mediapipeCamera.current.stop();
        webcamVideo.current = null;
    }

    useEffect(() => {
        if(dynamic) socket.on("dynamic", data => onResult(parseData(data)))
        else socket.on("stream", data => onResult(parseData(data)))
    }, [dynamic])

    useEffect(() => {
        async function initCamera() {
            mediapipeCamera.current = new Camera(webcamVideo.current, {
              onFrame: async () => {
                if(webcamVideo.current) await hands.current.send({ image: webcamVideo.current });
              },
            })
            mediapipeCamera.current.start();
        }

        initCamera();
        loadHands();
    }, [])

    return {
        captureState,
        setCaptureState,
        webcamVideoRef: webcamVideo,
        teardown
    }
}

export default useWebcam;