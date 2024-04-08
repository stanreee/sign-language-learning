import { useState } from 'react'
import Webcam from '../components/Webcam' 
import '../styles/Practice.css'
import '../styles/Quiz.css'
import React from 'react'
import ASLLetters from "../images/ASLLetters.png"
import { useNavigate } from 'react-router-dom'
import Modal from 'react-modal';
import Words from '../components/ContainerLetters/Words'
import PracticeHelpModal from '../components/Modals/PracticeHelpModal'
import QuestionWords from '../components/ContainerLetters/QuestionWords'

const Practice = () => {

    const [result, setResult] = useState("A")
    // const [dynamic, setDynamic] = useState(false)
    const [confidence, setConfidence] = useState("")
    const [modalIsOpen, setModalOpen] = useState(true);
    const [wordsModalOpen, setWordsModalOpen] = useState(false);
    const [questionWordsModalOpen, setQuestionWordsModalOpen] = useState(false);

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
                <div> You can practice the signs you have learned here! </div>
                <li> Remember to keep your hands in the webcam frame at all times (indicated by green outline) </li>
                <li> Ensure that your camera is level with your hands </li>
                <li> When performing signs without motion, use the "Static" button </li>
                <li> When performing signs with motion, use the "Dynamic" button </li>  
                <li> For dynamic signs, select the type of sign you want to practice (1 vs 2 Handed) </li>    
                <li> After the type has been selected, hit the "Start Recording" button, then perform the sign you want after the countdown </li>          
            </div>
        )}
    const options = () => {
        return (
            <div style={{textAlign: "left"}}>
                <h2> Static (Non-Moving) Sign Options: </h2>
                <div> All Letters EXCEPT for J & Z </div>
                <h2> Dynamic (Moving) Sign Options: </h2>
                <h3> 1 Hand: </h3>
                <div> Letters = J, Z </div>
                <div> Basic Words/Phrases = Hello, Please, Thank You, </div>
                <div> Yes, No, Need, Home, Future </div>
                <div> Question Words = Who?, Where? Why? </div>
                <h3> 2 Hands: </h3>
                <div> Basic Words/Phrases = Family, Friend, Spaghetti </div>
                <div> Question Words = What?, When?, How? </div>
            </div>
        )}

    const modalStyles = {
        content: {
            height: "fit-content",
            width: "50%",
            right: "auto",
            bottom: "auto",
            top: "50%",
            left: "50%",
            transform: 'translate(-50%, -50%)',
            borderRadius: "10px",
        }
    }

    const openModal = () => {
        setModalOpen(true);
    }

    const closeModal = () => {
        setModalOpen(false);
    }

    return(
        <div className="practice-page">
            <Modal
                isOpen={modalIsOpen}
                onRequestClose={closeModal}
                style={modalStyles}
            >
                <div className="modal-header">
                    <h2>
                        Instructions
                    </h2>
                </div>
                <div className="modal-content">
                    {instructions()}
                </div>
                <div className="modal-footer">
                    <button className="Button" onClick={closeModal}>Continue</button>
                </div>
            </Modal>
            <PracticeHelpModal 
                isOpen={wordsModalOpen}
                onRequestClose={() => setWordsModalOpen(false)}
                title="Basic Words/Phrases"
            >
                <Words className="words-modal"/>
            </PracticeHelpModal>

            <PracticeHelpModal 
                isOpen={questionWordsModalOpen}
                onRequestClose={() => setQuestionWordsModalOpen(false)}
                title="Question Words"
            >
                <QuestionWords className="words-modal"/>
            </PracticeHelpModal>

            <div className='practice-page-col'>
                <div className="practice-header">
                    <h1 style={{marginRight: "24px"}}>Practice</h1>
                    <button className="Button" onClick={openModal}>Instructions</button>
                </div>
                <div>
                    {/* <div>{instructions()}</div> */}

                    <br />
                    <div className='letterAlign'>
                        <div className="center">
                            <br />
                            <div>
                            <div> Go to <a style={{cursor: "pointer", color: "#003459", fontWeight: "bold"}} onClick={navigateToLearn_letters}>Learning Chapters</a> to Refresh </div>
                            <div>Your Memory or Learn More!</div>
                            <div style={{display: "flex", marginTop: "12px", marginBottom: "12px"}}>
                                <button style={{marginRight: "12px"}} className={"Button"} onClick={() => setWordsModalOpen(true)}>Basic Words/Phrases</button>
                                <br />
                                <button className={"Button"} onClick={() => setQuestionWordsModalOpen(true)}>Question Words</button>
                            </div>
                            <div id="ref_image" data-hs-anchor="true" ><img src={ASLLetters} width={763.5} height={343.5} alt="aslLetters"/></div>
                        </div>
                        </div>
                        {/* <br /> */}

                        {/* <br /> */}
                        <div className="webcam-container">
                            <div>
                                <div>
                                    <span className="result-prompt">Result: </span>
                                    <span className="result">{result}</span>
                                    <br />
                                    <span className="result-prompt small">Confidence: </span>
                                    <span className="result small">{Math.round(Number(confidence)*100).toFixed(1)}%</span>
                                </div>
                            </div>
                            <Webcam canChangeType={true} hands={1} text={result} setText={setResult} setConfidence={setConfidence} isDynamic={false}/>
                        </div>
                    </div>
                    
                </div>
                
            </div>          
        </div>
    )
}

export default Practice;