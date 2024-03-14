/* eslint-disable prettier/prettier */
import express from 'express';
import dotenv from 'dotenv';
import mongoose from 'mongoose';
// import routes from "./routes/routes.js";

import Item from "./models/Item";
// const Item = require("./models/Item"); // Create the Item model

// const mongoString = "mongodb+srv://CassieB:Azxsw21@aslingo.4cr3x99.mongodb.net/?retryWrites=true&w=majority&appName=ASLingo"
mongoose.connect("mongodb://localhost/ASLingo", {
  useNewUrlParser: true,
  useUnifiedTopology: true,
});

app.get("/api/items", async (req, res) => {
  try {
    const items = await Item.find();
    res.json(items);
  } catch (error) {
    console.error(error);
    res.status(500).send("Server Error");
  }
});

// mongoose.connect(mongoString);
const database = mongoose.connection

dotenv.config({ path: './config/config.env' });

database.on('error', (error) => {
    console.log(error)
})

database.once('connected', () => {
    console.log('Database Connected');
})

const app = express();

app.use(express.json());
// app.use('/api', routes)

app.get('/', (req, res) => res.send('Server running'));

const PORT = process.env.PORT || 5000;

// app.listen(PORT, console.log(`Server running in ${process.env.NODE_ENV} mode on port ${PORT}`));
app.listen(PORT, () => console.log(`Server running on port ${PORT}`));