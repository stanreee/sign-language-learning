import { Link } from "react-router-dom"
import Words from "./ContainerLetters/Words"
import QuestionWords from "./ContainerLetters/QuestionWords"

const DisplayWords = () => {
  return (
    <>
    <div className="container" style={{textAlign: "left"}}>
      <div>
        <Link 
          to={`https://www.handspeak.com/word/`}
          style={{color: '#007EA7'}}
          > <div > Click here to learn more about signing different words from Handspeak! </div>
        </Link>
      </div>
      <div>
        <Link 
        to={`https://lifeprint.com/`}
        style={{color: '#007EA7'}}
        > <div > Or click here for more from Lifeprint! </div>
        </Link>
      </div>
    </div>

    <br />

    <div>Some Tips for Signing Words</div>
    <div className="container" style={{textAlign: "left"}}>
      <li> 
        Unlike most of the letters, these signs are dynamic so you move your hands when signing.
      </li>
      <li>
        The photos show a progression of what actions each sign involves. 
      </li>
      <li>
        Watch the videos for each sign, they are very helpful!
      </li>
      
    </div>    

    <Words />

    <br />

    <div id="question_words" data-hs-anchor="true">Some Tips for Signing Question Words</div>
    <div className="container" style={{textAlign: "left"}}>
      <li> 
        When signing question words like the ones shown here,
        you should furrow your brows. This is helpful since a 
        lot of ASL is not just shown through your hand movements, 
        but your expressions as well. 
      </li>
      <li>
        Just like you would raise your voice at the end of a question in English, 
        you furrow your brows to indicate you're asking a question. 
      </li>
      <li>
        Watch the videos for each sign, they are very helpful!
      </li>
      
    </div>

    <QuestionWords />

    <br />
    </>
  );
};

export default DisplayWords;
