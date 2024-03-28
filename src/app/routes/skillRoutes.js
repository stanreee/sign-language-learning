import express from 'express';
const router = express.Router();
import SkillController from '../controllers/skillController.js';

/* GET users listing. */
router.get('/', SkillController.getAllSkill);

router.post('/get-skill', SkillController.getOneSkill);

router.post('/get-level', SkillController.getOneLevel);

router.post('/update-skill', SkillController.updateOneSkill);

router.post('/post-quiz', SkillController.postQuiz);

export default router;
