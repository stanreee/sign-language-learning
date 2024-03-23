/* eslint-disable prettier/prettier */
import express from 'express';
import mongoose from 'mongoose';
import usersRouter from './routes/userRoutes.js';
import skillRouter from './routes/skillRoutes.js';
import cors from 'cors';
import 'dotenv/config'


const app = express();

app.use(express.json());
app.use(cors());
app.use(express.urlencoded({ extended: true }));

const port = process.env.PORT || 5001;

// Database Connection Process 
//REPLACE WITH CONNECTION STRING
const connectionString = process.env.DATABASE_URL;

//mongoose.connect(mongoString);
(async () => {
  try {
    await mongoose.connect(connectionString);
    console.log('connected successfully');
  } catch (err) {
    console.log('mongodb connection error: ' + err)
  }
})()

app.use("/users", usersRouter);
app.use("/skills", skillRouter);

app.get('/', (req, res) => {
  res.send('Hello World!');
})

app.listen(port, () => {
  console.log(`Example app listening on port ${port}`)
})
