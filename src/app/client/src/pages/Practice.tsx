import { useEffect, useRef, useState } from 'react'
import Webcam from '../components/Webcam' 
import '../styles/Practice.css'
import '../styles/Quiz.css'

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
                <div className='practice-webcam'>
                    <Webcam text={result} setText={setResult}/>
                </div>
            </div>          
        </div>
    )
}

export default Practice;