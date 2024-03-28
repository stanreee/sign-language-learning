import SkillService from '../services/skillService.js';

// where business logic is, and then could send to userservice or output

const getAllSkill = async (req, res) => {
  const allSkills = await SkillService.getAllSkill();
  res.send(allSkills);
};

const getOneSkill = async (req, res) => {
  if (req.body.email !== undefined && req.body.email !== '') {
    const email = req.body.email;

    //const userId = req.params.UserId;
    console.log(email);
    const Skill = await SkillService.getOneSkill(email);
    console.log(Skill);


    // const level = 0;
    // if (Skill !== undefined) {
    //   const level = parseInt(Math.floor(Skill[0].correctQuestions / 10));
    //   console.log(Skill[0].correctQuestions, level);
    // }
    if (Skill !== undefined || Skill.length != 0) {
      const resp = {
        attemptedQuestions: Skill[0].attemptedQuestions,
        correctQuestions: Skill[0].correctQuestions,
        level: parseInt(Math.floor(Skill[0].correctQuestions / 10))
      };
      res.send(resp);
    } else {
      const resp = {
        attemptedQuestions: 0,
        correctQuestions: 0,
        level: 0
      };
      res.send(resp);
    }
  } else {
    res.send({ error: 'Please pass request body with email' });
  }
};

const getOneLevel = async (req, res) => {
  if (req.body.email !== undefined) {
    const email = req.body.email;

    //const userId = req.params.UserId;
    console.log('email: ', email);
    const Skill = await SkillService.getOneSkill(email);
    console.log(Skill);
    const level = parseInt(Math.floor(Skill[0].correctQuestions / 10));
    console.log(Skill[0].correctQuestions, level);
    res.send({ level: level });
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

const postQuiz = async (req, res) => {
  if (req.body !== undefined) {
    const { email, results } = req.body;

    console.log(results);

    let correctAns = 0;

    for (let i = 0; i < results.length; i++) {
      if (results[i].isCorrect) {
        correctAns += 1;
      }
    }

    const updateSkill = await SkillService.updateOneSkill(email, results.length, correctAns);
    res.status(201).json(updateSkill);
  } else {
    res.send({ error: 'Please pass request body' });
  }
};

export default {
  getAllSkill,
  getOneSkill,
  updateOneSkill,
  getOneLevel,
  postQuiz
};
