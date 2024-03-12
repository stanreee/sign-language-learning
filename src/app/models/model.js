/* eslint-disable prettier/prettier */
import mongoose from 'mongoose';

const dataSchema = new mongoose.Schema({
    name: {
        required: true,
        type: String
    },
    username: {
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
})

module.exports = mongoose.model('Data', dataSchema)