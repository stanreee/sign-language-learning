import { useEffect, useRef, useState } from 'react'
import Webcam from '../components/Webcam' 
import '../styles/Practice.css'
import '../styles/Quiz.css'
import React from 'react'
import ASLLetters from "../images/ASLLetters.png"

const Practice = () => {

    const [result, setResult] = useState("A")
    const [dynamic, setDynamic] = useState(false)
    const [confidence, setConfidence] = useState("")

    // console.log("result: " + result);
    return(
        <div className="practice-page">
            <div className='practice-page-col'>
                <div className="practice-header">
                    <h1>Practice</h1>
                </div>
                <div>
                    <div className="practice-header">
                        <div className="container-row">
                            {
                                dynamic ? (
                                    <div >
                                    <button className="Dyanmic-Button active" onClick={() => {setDynamic(false)}}>Static</button>
                                    <button className="Dyanmic-Button">Dynamic</button>
                                    </div>
                                ) : (
                                    <div>
                                    <button className="Dyanmic-Button" >Static</button>
                                    <button className="Dyanmic-Button active" onClick={() => {setDynamic(true)}}>Dynamic</button>
                                    </div>
                                    )

                            }
                        </div>
                    </div>
                    <div>
                        <span className="result-prompt">Result: </span>
                        <span className="result">{result}</span>
                    </div>
                    <div>
                        <span className="result-prompt small">Confidence: </span>
                        <span className="result small">{confidence}%</span>
                    </div>
                    <div >
                        <div className='letterAlign'>
                            <Webcam text={result} setText={setResult} setConfidence={setConfidence} isDynamic={dynamic}/>
                            <img src={ASLLetters} width={500} height={500} alt="aslLetters" />
                        </div>
                    </div>
                </div>
                
            </div>          
        </div>
    )
}

export default Practice;