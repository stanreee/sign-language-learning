import mongoose from 'mongoose';
const { Schema, model } = mongoose;

const userSchema = new Schema({
  // _id is the userId column in a booking
  name: {
    required: true,
    type: String
  },
  email: {
    required: true,
    type: String
  },
  password: {
    required: true,
    type: String
  },
  handedness: {
    required: true,
    type: String
  }
});

const UserModel = model('Users', userSchema);
UserModel.collection.name = 'users';
export default UserModel;
