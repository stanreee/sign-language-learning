import '../styles/Exercise.css';
import DisplayLetters from "../components/DisplayLetters"

const LearnLetters = () => {

    return (
    <div className="Exercise-Page">
      <div className="Exercise-header">
        <h1>Learn All The Letters!</h1>
      </div>
      <div className="Exercise-content">
        <h3>
            <DisplayLetters />
        </h3>
      </div>
    </div>
    );
}

export default LearnLetters;
