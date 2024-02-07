import iQuiz from "./iQuiz";
import iQuizASL from "./iQuizASL";

export default interface iQuizHard extends iQuiz {
    topic: string,
    level: "Hard",
    description: string,
    totalQuestions: 6,
    timePerQuestion: 10,
    questions: iQuizASL[] 
}