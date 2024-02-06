import { useEffect, useRef, useState } from 'react'
import Webcam from '../components/Webcam' 
import '../styles/Practice.css'
import '../styles/Quiz.css'
import React from 'react'
import ASLLetters from "../images/ASLLetters.png"

const Practice = () => {

    const [result, setResult] = useState("A")

    console.log("result: " + result);
    return(
        <div className="practice-page">
            <div className='practice-page-col'>
                <div className="practice-header">
                    <h1>Practice</h1>
                </div>
                <div>
                    <span className="result-prompt">Result: </span>
                    <span className="result">{result}</span>
                </div>
                <div className='letterAlign'>
                    <Webcam text={result} setText={setResult} run={true}/>
                    <img src={ASLLetters} width={500} height={500} alt="aslLetters" />
                </div>
                
            </div>          
        </div>
    )
}

export default Practice;