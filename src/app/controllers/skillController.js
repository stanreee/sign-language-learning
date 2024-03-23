import SkillService from '../services/skillService.js';

// where business logic is, and then could send to userservice or output

const getAllSkill = async (req, res) => {
  const allSkills = await SkillService.getAllSkill();
  res.send(allSkills);
};

const getOneSkill = async (req, res) => {
  if (req.body.email !== undefined) {
    const email = req.body.email;

    //const userId = req.params.UserId;
    console.log(email);
    const Skill = await SkillService.getOneSkill(email);
    console.log(Skill);
    res.send(Skill);
  } else {
    res.send({ error: 'Please pass request body with email' });
  }
};

const updateOneSkill = async (req, res) => {
  if (req.body !== undefined) {
    let { email, attemptedQuestions, correctQuestions } = req.body;

    console.log(email, attemptedQuestions, correctQuestions);
    const updateSkill = await SkillService.updateOneSkill(
      email,
      attemptedQuestions,
      correctQuestions
    );
    res.status(201).json(updateSkill);
  } else {
    res.send({ error: 'Please pass request body' });
  }
};

export default {
  getAllSkill,
  getOneSkill,
  updateOneSkill
};
