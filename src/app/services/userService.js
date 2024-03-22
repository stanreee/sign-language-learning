import UserModel from '../models/userModel.js';

import mongoose from 'mongoose';

const getAllUsers = async () => {
  console.log('Get all Users call');

  try {
    const all = await UserModel.find({});
    return all;
  } catch (error) {
    console.log(error);
    console.log('Users could not be found');
  }
};

const getOneUser = async (id) => {
  console.log(`Getting User ${id}`);

  try {
    const all = await UserModel.find({ email: id }).exec();
    return all;
  } catch (error) {
    console.log(error);
    console.log(`User ${id} could not be found!!`);
    return { error: 'no user found' };
  }
};

const addOneUser = async (requestData) => {
  console.log('Add one user');

  const name = requestData.name;
  const email = requestData.email;
  const password = requestData.password;
  const handedness = requestData.handedness;

  try {
    let user = new UserModel({
      name: name,
      email: email,
      password: password,
      handedness: handedness
    });

    user
      .save()
      .then(() => {
        console.log('User created successfully');
      })
      .catch((error) => {
        console.error('Error saving user:', error);
      });

    return { status: 200, user };
  } catch (error) {
    console.log(error);
    console.log('User could not be added');
  }
};

const deleteUser = async (id) => {
  console.log(`Delete User ${id}`);

  try {
    const deleteUser = await UserModel.deleteOne({ _id: new mongoose.Types.ObjectId(id) });
    return deleteUser;
  } catch (error) {
    console.log(error);
    console.log(`User ${id} could not be deleted`);
    return { error: 'user not deleted' };
  }
};

export default {
  getAllUsers,
  getOneUser,
  addOneUser,
  deleteUser
};
