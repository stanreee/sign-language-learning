import express from 'express';
const router = express.Router();
import SkillController from '../controllers/skillController.js';

/* GET users listing. */
router.get('/', SkillController.getAllSkill);

router.post('/get-skill', SkillController.getOneSkill);

router.post('/update-skill', SkillController.updateOneSkill);

export default router;
