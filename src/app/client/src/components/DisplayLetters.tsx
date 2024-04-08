import { Link } from "react-router-dom"
import Letters from "./ContainerLetters/Letters";

const DisplayLetters = () => {
  return (
    <>
      Some Tips for Fingerspelling
      <div className="container" style={{textAlign: "left"}}>
        <li>Hold the hand your signing with beside your cheek.</li>
        <li>
            Make sure that your hands are stationary, meaning that you do not "bounce" your hand between letters.
            This will make it harder for others to understand what you are spelling.  
        </li>
        <li>
          If you are signing two of the same letters in a row, you can "slide" your hand to the side when 
          repeating the second letter, or "bounce" your hand to indicate you are repeating the letter. 
        </li>

        <br />
        <div>
          <Link 
            to={`https://www.handspeak.com/learn/213/`}
            style={{color: '#007EA7'}}
            > <div > Click here to learn more about fingerspelling techniques from Handspeak! </div>
          </Link>
        </div>
        <div>
          <Link 
          to={`https://asl.ms/`}
          style={{color: '#007EA7'}}
          > <div > Or click here to test your letter recognition skills! </div>
          </Link>
        </div>
      </div>

      <Letters />
    </>
  );
};

export default DisplayLetters;
