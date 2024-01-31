import iQuizQuestions from "../interfaces/iQuizQuestions";

type ExerciseCardProps = {
    title: string;
    desc: string;
    difficulty: string;
    questionAmount: number;
    quiz: iQuizQuestions[];
    selected?: boolean;
    onClick?: () => void;
  };

const ExerciseCard = ({
    title,
    desc,
    difficulty,
    questionAmount,
    quiz,
    selected = false,
    onClick
  }: ExerciseCardProps) => {
    return (
        <div className="Exercise-Card">
            <div>
            <h3>{title}</h3>
            <p>{desc}</p>
            <p>Difficulty: {difficulty}</p>
            <p>Questions: {questionAmount}</p>
                <button className="Exercise-Begin-Button" onClick={onClick}> Start</button>
            </div>
        </div>
    );
}

export default ExerciseCard;
