import SkillModel from '../models/skillModel.js';

const getAllSkill = async () => {
  console.log('Get all Users call');

  try {
    const all = await SkillModel.find({});
    return all;
  } catch (error) {
    console.log(error);
    console.log('Users could not be found');
  }
};

const getOneSkill = async (email) => {
  console.log(`Getting User ${email}`);

  try {
    const all = await SkillModel.find({ email: email }).exec();
    return all;
  } catch (error) {
    console.log(error);
    console.log(`User ${email} could not be found!!`);
    return { error: 'no user found' };
  }
};

const updateOneSkill = async (email, attemptedQuestions, correctQuestions) => {
  try {
    const confirmed = await SkillModel.updateOne(
      {
        email: email
      },
      {
        $inc: {
          attemptedQuestions: attemptedQuestions,
          correctQuestions: correctQuestions
        }
      }
    );

    return confirmed;
  } catch (error) {
    console.log(error);
    console.log('Creation hash does not exist');
  }
};

const addOneSkill = async (email) => {
  console.log('Add one skill');

  try {
    let skill = new SkillModel({
      email: email,
      attemptedQuestions: 0,
      correctQuestions: 0
    });

    skill
      .save()
      .then(() => {
        console.log('Skill created successfully');
      })
      .catch((error) => {
        console.error('Error saving skill:', error);
      });

    return { status: 200, skill };
  } catch (error) {
    console.log(error);
    console.log('User could not be added');
  }
};

export default {
  getAllSkill,
  getOneSkill,
  updateOneSkill,
  addOneSkill
};
