import iQuiz from "./iQuiz";
import iQuizASL from "./iQuizASL";

export default interface iQuizMedium extends iQuiz {
    topic: string,
    level: "Medium",
    description: string,
    totalQuestions: 5,
    timePerQuestion: 15,
    questions: iQuizASL[] 
}