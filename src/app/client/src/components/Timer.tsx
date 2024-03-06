import React, { useEffect, useState } from 'react';
import { useTimer } from 'react-timer-hook';
import '../styles/Timer.css'

// from : https://www.npmjs.com/package/react-timer-hook

const Timer = ({ time, setIsExpired, timerRes }: {time: number, setIsExpired: React.Dispatch<React.SetStateAction<boolean>>, timerRes: boolean}) => {
    
    const [reset, setReset] = useState<boolean>(false);
    
    const expiryTimestamp = new Date();
    expiryTimestamp.setSeconds(expiryTimestamp.getSeconds() + time); // 1 minutes timer
  
    const {
        totalSeconds,
        seconds,
        minutes,
        hours,
        days,
        isRunning,
        start,
        pause,
        resume,
        restart,
    } = useTimer({ expiryTimestamp, onExpire: () => {
        setIsExpired(true); 
        setReset(true);
        }
    });

    const restartTimer = () => {
        const restartTime = new Date();
        restartTime.setSeconds(restartTime.getSeconds() + time);
        console.log('here');
        restart(restartTime, true);
    }

    useEffect(() => {
        if(reset){
            setReset(false);
            restartTimer();
            setIsExpired(true); 
        }
    }, [reset])

    useEffect(() => {
        if(timerRes){
            setReset(false);
            restartTimer();
        }
    }, [timerRes])

    //console.log('here');

    return (
        <div >
        {/* <h1>react-timer-hook </h1>
        <p>Timer Demo</p> */}
        <div className='timer-display'>
            <span>{minutes}</span>:<span>{seconds > 9 ? seconds : '0' + seconds}</span>
        </div>
        {/*<p>{isRunning ? 'Running' : 'Not running'}</p>
        <button onClick={start}>Start</button>
        <button onClick={pause}>Pause</button>
        <button onClick={resume}>Resume</button> 
        <button onClick={restartTimer}>Restart</button>*/}
        </div>
    );
}

export default Timer