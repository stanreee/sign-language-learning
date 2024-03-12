import { createContext, useCallback, useEffect, useRef, useState } from "react";
import { io, Socket } from "socket.io-client";
import getFeatures from "../util/getFeatures";
import { Hands } from "@mediapipe/hands";
import { Camera } from '@mediapipe/camera_utils';

const LIMIT_FPS = 15;

function throttle(callback, limit) {
    let waiting = false;
    return function (...args) {
        if (!waiting) {
          callback.apply(this, args);
          waiting = true;
          setTimeout(function () {
            waiting = false;
          }, limit);
        }
    };
}

function useWebcam({
    numHands,
    dynamic,
    onCaptureError,
    handedness, // for left hand folks
    onResult,
}) {
    const landmarkHistory = useRef([]);
    const [captureState, setCaptureState] = useState(false);
    const socket = io("http://127.0.0.1:5000");
    const hands = useRef(null);
    const webcamVideo = useRef(null);
    const mediapipeCamera = useRef(null),
    SocketContext = createContext<Socket>(socket);

    const onResults = (results) => {
        console.log("capturing");
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
        if(hands.current) hands.current.onResults(throttle(onResults, LIMIT_FPS));
        if(captureState && !dynamic) {
            throw new Error("useWebcam cannot start capture if webcam is not dynamic");
        }
    }, [captureState])

    const loadHands = () => {
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
        hands.current.onResults(throttle(onResults, LIMIT_FPS));
    }

    const parseData = (data) => {
        const deserialized = JSON.parse(data);
        return deserialized;
    }

    useEffect(() => {
        async function initCamera() {
            mediapipeCamera.current = new Camera(webcamVideo.current, {
              onFrame: async () => {
                await hands.current.send({ image: webcamVideo.current });
              },
            })
            mediapipeCamera.current.start();
        }

        if(dynamic) socket.on("dynamic", data => onResult(parseData(data)))
        else socket.on("stream", data => onResult(parseData(data)))

        initCamera();
        loadHands();
    }, [dynamic])

    return {
        captureState,
        setCaptureState,
        webcamVideoRef: webcamVideo
    }
}

export default useWebcam;