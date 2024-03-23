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
    const instructions = () => {
        return (
            <div>
                <h3> Instructions: </h3>
                <div> You can practice the signs you have learned here! </div>
                <li> Remember to keep your hands in the webcam frame at all times </li>
                <li> When performing signs without motion, use the "Static" button </li>
                <li> When performing signs with motion, use the "Dynamic" button </li>  
                <li> For dynamic signs, hit the "Start Recording" button, then perform the sign you want to practice </li>              
            </div>
        )}
    const options = () => {
        return (
            <div style={{textAlign: "left"}}>
                <h3> Static (Non-Moving) Sign Options: </h3>
                <div> All Letters EXCEPT for J & Z </div>
                <h3> Dynamic (Moving) Sign Options: </h3>
                <div> 
                    Basic Words/Phrases = 
                    Hello, Please,
                    Thank You, Yes, No, 
                </div>
                <div>
                    Need,
                    Home, Family,
                    Friend, Future,
                    Spaghetti
                </div>
                <div>
                    Question Words = 
                    Who? What? Where?
                    When? Why? How? 
                </div>
            </div>
        )}

    // console.log("result: " + result);
    return(
        <div className="practice-page">
            <div className='practice-page-col'>
                <div className="practice-header">
                    <h1>Practice</h1>
                </div>
                <div>
                    <div>{instructions()}</div>

                    <br />
                    <div className='letterAlign'>
                        <br />
                        <div>
                            <div> Go to Learning Chapters to Refresh </div>
                            <div>Your Memory or Learn More!</div>
                            <br />
                            <a href="#ref_image" rel="noopener">Click for Letter Reference Image  </a>
                            <br />
                            <button className={"Dyanmic-Button"} onClick={navigateToLearn_letters}>Letters </button>
                            <br />
                            <button className={"Dyanmic-Button"} onClick={navigateToLearn_basic_words}>Basic Words/Phrases</button>
                            <br />
                            <button className={"Dyanmic-Button"} onClick={navigateToLearn_question_words}>Question Words</button>
                        </div>

                        <br />
                        <Webcam hands={1} text={result} setText={setResult} setConfidence={setConfidence} isDynamic={dynamic}/>

                        <div>
                            <div>
                                <span className="result-prompt">Result: </span>
                                <span className="result">{result}</span>
                                <br />
                                <span className="result-prompt small">Confidence: </span>
                                <span className="result small">{Math.round(Number(confidence)*100).toFixed(1)}%</span>
                            </div>
                            <br />
                            <div>
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
                            <br /> 
                            {options()}
                        </div>
                    </div>

                    <br />
                    <div style={{textAlign: "center"}}>                      
                        <br />
                        <div className='letterAlign' >
                            <br />
                            <div className="center">
                                <div id="ref_image" data-hs-anchor="true" ><img src={ASLLetters} width={763.5} height={343.5} alt="aslLetters"/></div>
                                <br />
                            </div>
                            <br />
                        </div>
                    </div>
                    
                </div>
                
            </div>          
        </div>
    )
}

export default Practice;