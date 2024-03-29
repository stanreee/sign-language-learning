import who from "../images/learnWords/who.png"
import what from "../images/learnWords/what.png"
import where from "../images/learnWords/where.png"
import when from "../images/learnWords/when.png"
import why from "../images/learnWords/why.png"
import how from "../images/learnWords/how.png"
import hello from "../images/learnWords/hello.png"
import please from "../images/learnWords/please.png"
import thanku from "../images/learnWords/thanku.png"
import yes from "../images/learnWords/yes.png"
import no from "../images/learnWords/no.png"
import need from "../images/learnWords/need.png"
import home from "../images/learnWords/home.png"
import family from "../images/learnWords/family.png"
import friend from "../images/learnWords/friend.png"
import future from "../images/learnWords/future.png"
import spaghetti from "../images/learnWords/spaghetti.png"
import youtube from "../images/learnLetters/youtube.png"

import wordsResource from "../data/ASLSigns.json"
import { Link } from "react-router-dom"

const wordCard = (num: string, index: number, width: number, height: number) => {
    // DATA CALL
    const allWords = wordsResource.words;
    const wordsDesc = wordsResource.wordsDesc;
    const wordVideos = wordsResource.wordVideos;

    return (
      <div className = "card">
        <img src={num} width={width} height={height} alt= {`${allWords[index]}`} />
        <h2>{allWords[index]}</h2>
        {wordsDesc[index]}
        <Link 
          to={`https://youtu.be/${wordVideos[index]}`}
          style={{color: '#000000',}}
          > <div > <img src={youtube} width={46.2} height={32.4} alt= "youtube" /> </div>
        </Link>
      </div>
    )
}

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

    <div className="container-letters">
      <div id="hello" data-hs-anchor="true">{wordCard(hello, 6, 370, 270)}</div>               
      <div id="please" data-hs-anchor="true">{wordCard(please, 7, 270, 270)}</div>      
      <div id="thanku" data-hs-anchor="true">{wordCard(thanku, 8, 570, 270)}</div>      
      <div id="yes" data-hs-anchor="true">{wordCard(yes, 9, 300, 270)}</div>         
      <div id="no" data-hs-anchor="true">{wordCard(no, 10, 570, 270)}</div>       
      <div id="need" data-hs-anchor="true">{wordCard(need, 11, 570, 270)}</div>       
      <div id="home" data-hs-anchor="true">{wordCard(home, 12, 670, 270)}</div>  
      <div id="family" data-hs-anchor="true">{wordCard(family, 13, 370, 270)}</div> 
      <div id="friend" data-hs-anchor="true">{wordCard(friend, 14, 300, 300)}</div> 
      <div id="future" data-hs-anchor="true">{wordCard(future, 15, 670, 270)}</div> 
      <div id="spaghetti" data-hs-anchor="true">{wordCard(spaghetti, 16, 370, 270)}</div> 
    </div>

    <div>
      Jump to:
      <br />
      <a href="#hello" rel="noopener">Hello,   </a>
      <a href="#please" rel="noopener">Please,  </a>
      <a href="#thanku" rel="noopener">Thank You   </a>
      <br />
      <a href="#yes" rel="noopener">Yes,   </a>
      <a href="#no" rel="noopener">No,   </a>
      <a href="#need" rel="noopener">Need,   </a>
      <a href="#home" rel="noopener">Home   </a>
      <br />
      <a href="#family" rel="noopener">Family,   </a>
      <a href="#friend" rel="noopener">Friend,   </a>
      <a href="#future" rel="noopener">Future,   </a>
      <a href="#spaghetti" rel="noopener">Spaghetti   </a>
    </div>

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

    <div className="container-letters">
      <div id="who" data-hs-anchor="true">{wordCard(who, 0, 370, 270)}</div>
      <div id="what" data-hs-anchor="true">{wordCard(what, 1, 270, 270)}</div>
      <div id="where" data-hs-anchor="true">{wordCard(where, 2, 370, 270)}</div>
      <div id="when" data-hs-anchor="true">{wordCard(when, 3, 370, 270)}</div>
      <div id="why" data-hs-anchor="true">{wordCard(why, 4, 370, 270)}</div>
      <div id="how" data-hs-anchor="true">{wordCard(how, 5, 670, 270)}</div>       
    </div>

    <div>
      Jump to:
      <br />
      <a href="#who" rel="noopener">Who?   </a>
      <a href="#what" rel="noopener">What?   </a>
      <a href="#where" rel="noopener">Where?   </a>
      <a href="#when" rel="noopener">When?   </a>
      <a href="#why" rel="noopener">Why?   </a>
      <a href="#how" rel="noopener">How?   </a>
    </div>

    <br />
    </>
  );
};

export default DisplayWords;
