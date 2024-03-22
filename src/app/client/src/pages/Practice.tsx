import { useEffect, useRef, useState } from 'react'
import Webcam from '../components/Webcam' 
import '../styles/Practice.css'
import '../styles/Quiz.css'
import React from 'react'
import ASLLetters from "../images/ASLLetters.png"

// Call the server API to check if the given email ID already exists
const mockPutInDB = () => {

    const userId = "nRR0O9xPLO"

    // fetch('http://localhost:5001/get-stats', {
    // method: 'POST',
    // headers: {
    //     'Content-Type': 'application/json',
    // },
    // body: JSON.stringify({userId}),
    // })
    // .then((r) => r.json())
}

const Practice = () => {

    const [result, setResult] = useState("A")

    
    // console.log("result: " + result);
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
                <div>
                    <button onClick={mockPutInDB}> PRESS</button>
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