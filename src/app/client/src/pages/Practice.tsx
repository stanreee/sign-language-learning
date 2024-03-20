import { useState } from 'react'
import Webcam from '../components/Webcam' 
import '../styles/Practice.css'
import '../styles/Quiz.css'
import React from 'react'
import ASLLetters from "../images/ASLLetters.png"
import { useNavigate } from 'react-router-dom'

const Practice = () => {

    const [result, setResult] = useState("A")
    const [dynamic, setDynamic] = useState(false)
    const [confidence, setConfidence] = useState("")

    const navigate = useNavigate();
    const navigateToLearn_letters = () => {
        navigate("/learn");
        setTimeout(() => {
        const contactSection = document.getElementById("letters");
        if (contactSection) {
        contactSection.scrollIntoView({ behavior: "smooth" });
        }}, 100); // Delay for smoother scroll
        return
    }
    const navigateToLearn_basic_words = () => {
        navigate("/learn");
        setTimeout(() => {
        const contactSection = document.getElementById("basic_words");
        if (contactSection) {
        contactSection.scrollIntoView({ behavior: "smooth" });
        }}, 100); // Delay for smoother scroll
        return
    }
    const navigateToLearn_question_words = () => {
        navigate("/learn");
        setTimeout(() => {
        const contactSection = document.getElementById("question_words");
        if (contactSection) {
        contactSection.scrollIntoView({ behavior: "smooth" });
        }}, 100); // Delay for smoother scroll
        return
    }

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
                        <span className="result small">{Math.round(Number(confidence)*100).toFixed(1)}%</span>
                    </div>
                    <div >
                        <div className='letterAlign'>
                            <Webcam text={result} setText={setResult} setConfidence={setConfidence} isDynamic={dynamic}/>
                            <img src={ASLLetters} width={763.5} height={343.5} alt="aslLetters" />
                        </div>
                    </div>
                    <br />
                    <div style={{textAlign: "left"}}>
                        Go to Learning Chapters:
                        <br />
                        <button className={"Dyanmic-Button"} onClick={navigateToLearn_letters}>Letters </button>
                        <button />
                        <button className={"Dyanmic-Button"} onClick={navigateToLearn_basic_words}>Basic Words/Phrases</button>
                        <button />
                        <button className={"Dyanmic-Button"} onClick={navigateToLearn_question_words}>Question Words</button>
                    </div>
                </div>
                
            </div>          
        </div>
    )
}

export default Practice;