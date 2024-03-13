/* eslint-disable prettier/prettier */
import mongoose from 'mongoose';

const { Schema, model } = mongoose;

const usersSchema = new Schema({
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

const Users = model('Users', usersSchema);
export default Users;