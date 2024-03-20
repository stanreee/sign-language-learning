import { Link } from 'react-router-dom';
import DisplayLetters from "../components/DisplayLetters"
import DisplayWords from '../components/DisplayWords'
import info from "../data/fyInfo.json"


const Learn = () => {

    //DATA CALL
    const NADTitle = info.NADHeader; 
    const NADDesc = info.NADDesc.join(" "); 
    const HSTitle = info.HSHeader;
    const HSDesc = info.HSDesc.join(" ");
    const gallTitle = info.gallaudetHeader;
    const gallDesc = info.gallaudetDesc.join(" ");

    const partsOfPage = () => {
      return (
        <div>
          Learning Chapters:
          <br />
          <a href="#letters" rel="noopener">Letters,   </a>
          <a href="#basic_words" rel="noopener">Basic Words/Phrases,   </a>
          <a href="#question_words" rel="noopener">Question Words,  </a>
          <a href="#add_resources" rel="noopener">Additional Resources   </a>
      </div>
    )}

    return (
    <div className="App">
      <header className="App-header">

        <div>
          {partsOfPage()}
        </div>

        <br />
        <h2 id="letters" data-hs-anchor="true">Learn the Letters!</h2>
        <br />
        <DisplayLetters />

        <br />

        <h2 id="basic_words" data-hs-anchor="true">Learn Some Basic Words/Phrases!</h2>
        <br />
        <DisplayWords />
      </header>

      <div className="Section"> </div>

      <header className="App-header2">
        <h1 id="add_resources" data-hs-anchor="true">Additional Resources</h1>
        <div>
          There are plenty of resources to learn more about ASL below!
          These are from trusted sources in the Deaf community and are 
          a great way to learn more about ASL from their perspectives. 
        </div>
      </header>

      <div className = 'Section-container' id = "additionalResources">
      <div className = "Section">
            <h2> {NADTitle} </h2>
            <p> {NADDesc} </p>
            <div>
                <Link 
                  to="https://www.nad.org/resources/american-sign-language/community-and-culture-frequently-asked-questions/"
                  style={{color: '#000',}}
                  >Go to NAD Website
                </Link>
            </div>
        </div>

        <div className = "Section">
            <h2> {HSTitle} </h2>
            <p> {HSDesc} </p>
            <div>
                <Link 
                  to="https://www.handspeak.com/"
                  style={{color: '#000',}}
                  >Go to HandSpeak Website
                </Link>
            </div> 
        </div>

        <div className = "Section">
            <h2> {gallTitle} </h2>
            <p> {gallDesc} </p>
            <div>
                <Link 
                  to="https://gallaudet.edu/about/#mission-vision#mission-vision"
                  style={{color: '#000',}}
                  >Go to Gallaudet University Website
                </Link>
            </div> 
        </div>
      </div>
    </div>
    );
}

export default Learn;
