import { createContext, useCallback, useEffect, useRef, useState } from "react";
import { io, Socket } from "socket.io-client";
import getFeatures from "../util/getFeatures";
import { Hands } from "@mediapipe/hands";
import { Camera } from '@mediapipe/camera_utils';

// var LIMIT_FPS = 30;

function useWebcam({
    numHands,
    onCaptureError,
    onHandDetection,
    handedness, // for left hand folks
    onResult,
    debug
}) {
    const landmarkHistory = useRef([]);
    const [captureState, setCaptureState] = useState(false);
    const [dynamic, setDynamic] = useState(false);
    const [recordingState, setRecordingState] = useState(0);
    const socket = io("http://127.0.0.1:5000");
    const hands = useRef(null);
    const webcamVideo = useRef(null);
    const mediapipeCamera = useRef(null),
    SocketContext = createContext<Socket>(socket);

    const date = new Date();

    const startTime = useRef(date.getTime() / 1000)
    // const prevTime = useRef(0);
    // const deltaTime = useRef(0);
    // const frameCount = useRef(0);
    // const totalFPS = useRef(0);

    const mediapipeFrameData = useRef({
        prevTime: 0,
        deltaTime: 0,
        frameCount: 0,
        totalFPS: 0,
        currentFPS: 0,
    })

    const frameTracker = useRef(-1);
    const frameTrackerDelayed = useRef(-50);
    // const totalWebcamFrames = useRef(0);

    // const currentFPS = useRef(0);

    const FPS_THROTTLE = useRef(30.0);

    // const [fpsRange, setFpsRange] = useState([1.5, 2.5])

    const fpsRange = useRef([1.5, 3.5])

    const throttle = useCallback((callback) => {
        if(debug) console.log("throttle", FPS_THROTTLE)
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

    function getMediapipeFrameData () {
        // let prevTime, deltaTime, frameCount, totalFPS, currentFPS;
        let { prevTime, deltaTime, frameCount, totalFPS, currentFPS } = mediapipeFrameData.current;
        if(prevTime === 0) {
            const newDate = new Date();
            prevTime = newDate.getTime() / 1000;
        }else {
            const newDate = new Date();
            const curTime = newDate.getTime() / 1000;
            const prevDeltaTime = curTime - prevTime;

            deltaTime = newDate.getTime() / 1000 - startTime.current;

            // const avgFPS = frameCount.current / deltaTime.current;
            currentFPS = 1 / prevDeltaTime;
            totalFPS = mediapipeFrameData.current.totalFPS + currentFPS;
            const avgFPS = totalFPS / mediapipeFrameData.current.frameCount;

            frameCount = mediapipeFrameData.current.frameCount + 1;
            prevTime = curTime;

            if(debug) console.log("FRAMERATE:", currentFPS, "AVG FPS:", avgFPS, "THROTTLE:", FPS_THROTTLE.current);
        }

        mediapipeFrameData.current = {
            prevTime,
            deltaTime,
            frameCount,
            totalFPS,
            currentFPS,
        }
    }

    const onResults = (results) => {
        getMediapipeFrameData();
        const { multiHandLandmarks, multiHandedness } = results;
        if(multiHandLandmarks.length >= numHands) {
            frameTrackerDelayed.current = frameTracker.current;
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
                    if(debug) console.log("RECORDING");
                    if(landmarkHistory.current.length < 30) {
                        landmarkHistory.current.push(totalHandFeatures);
                        setRecordingState(landmarkHistory.current.length/30);
                    }else {
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
    }, [dynamic, captureState, FPS_THROTTLE])

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
        if(dynamic) {
            socket.on("dynamic", data => onResult(parseData(data)))
            FPS_THROTTLE.current = 30.0;
            // setFpsRange([13.5, 14.5])
            fpsRange.current = [12.5, 15.5]
        }
        else {
            socket.on("stream", data => onResult(parseData(data)))
            FPS_THROTTLE.current = 200.0;
            // setFpsRange([4.5, 5.5])
            fpsRange.current = [1.5, 3.5]
        }
    }, [dynamic, captureState])

    useEffect(() => {
        async function initCamera() {
            mediapipeCamera.current = new Camera(webcamVideo.current, {
              onFrame: async () => {
                const { currentFPS } = mediapipeFrameData.current;
                if(currentFPS > fpsRange.current[1]) {
                    FPS_THROTTLE.current += 0.5;
                } else if(currentFPS < fpsRange.current[0]) {
                    FPS_THROTTLE.current -= 0.5;
                    if(FPS_THROTTLE.current < 0) FPS_THROTTLE.current = 0;
                }

                frameTracker.current += 1;

                if(frameTracker.current - frameTrackerDelayed.current > 10) {
                    onHandDetection(false);
                }else {
                    onHandDetection(true);
                }

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
        setDynamic,
        webcamVideoRef: webcamVideo,
        teardown,
        recordingState,
    }
}

export default useWebcam;