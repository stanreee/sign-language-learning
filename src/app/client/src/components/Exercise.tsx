
import iQuizASL from "../interfaces/iQuizASL";
import iQuizQuestions from "../interfaces/iQuizQuestions";

type ExerciseCardProps = {
    title: string;
    desc: string;
    difficulty: string;
    timePerQuestion: number;
    questionAmount: number;
    quiz: iQuizASL[];
    selected?: boolean;
    onClick?: () => void;
  };

const ExerciseCard = ({
    title,
    desc,
    difficulty,
    questionAmount,
    timePerQuestion,
    quiz,
    selected = false,
    onClick
  }: ExerciseCardProps) => {
    return (
        <div className="Exercise-Card">
            <div>
            <h3>{title}</h3>
            <p>{desc}</p>
                <button className="Exercise-Begin-Button" onClick={onClick}> Start</button>
            </div>
        </div>
    );
}

export default ExerciseCard;
