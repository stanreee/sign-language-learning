import mongoose from 'mongoose';
const { Schema, model } = mongoose;

const skillSchema = new Schema({
  // _id is the userId column in a booking
  email: {
    required: true,
    type: String
  },
  attemptedQuestions: {
    required: true,
    type: Number
  },
  correctQuestions: {
    required: true,
    type: Number
  }
});

const SkillModel = model('Skill', skillSchema);
SkillModel.collection.name = 'skill';
export default SkillModel;
