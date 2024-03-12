/* eslint-disable prettier/prettier */
import express from 'express';
import dotenv from 'dotenv';
import mongoose from 'mongoose';
import routes from "../routes/routes"

const mongoString = process.env.DATABASE_URL

mongoose.connect(mongoString);
const database = mongoose.connection

dotenv.config({ path: './config/config.env' });

database.on('error', (error) => {
    console.log(error)
})

database.once('connected', () => {
    console.log('Database Connected');
})

const app = express();

app.get('/', (req, res) => res.send('Server running'));

app.use(express.json());
app.use('/api', routes)

const PORT = process.env.PORT || 5000;

app.listen(PORT, console.log(`Server running in ${process.env.NODE_ENV} mode on port ${PORT}`));
