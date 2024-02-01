import { Link } from 'react-router-dom';
import LearnLetters from '../components/LearnLetters';
import info from "../data/fyInfo.json"

const Learn = () => {

    //DATA CALL
    const NADTitle = info.NADHeader; 
    const NADDesc = info.NADDesc.join(" "); 
    const HSTitle = info.HSHeader;
    const HSDesc = info.HSDesc.join(" ");
    const gallTitle = info.gallaudetHeader;
    const gallDesc = info.gallaudetDesc.join(" ");

    return (
    <div className="App">
      <header className="App-header">
        <LearnLetters /> 
      </header>

      <div className="Section"> </div>

      <header className="App-header2">
        <h1>Additional Resources</h1>
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
