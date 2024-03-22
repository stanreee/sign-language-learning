import UserService from '../services/userService.js';
import jwt from 'jsonwebtoken';
import bcrypt from 'bcrypt';
import userService from '../services/userService.js';

const jwtSecretKey = 'dsfdsfsdfdsvcsvdfgefg';

// where business logic is, and then could send to userservice or output

const getAllUsers = async (req, res) => {
  const allUsers = await UserService.getAllUsers();
  res.send(allUsers);
};

const getOneUser = async (req, res) => {
  if (req.body.email !== undefined) {
    const email = req.body.email;

    //const userId = req.params.UserId;
    console.log(email);
    const User = await UserService.getOneUser(email);
    console.log(User);
    res.send(User);
  } else {
    res.send({ error: 'Please pass request body with email' });
  }
};

const addNewUser = async (req, res) => {
  if (req.body !== undefined) {
    let requestData = req.body;

    const newUser = await UserService.addOneUser(requestData);
    res.status(201).json(newUser);
  } else {
    res.send({ error: 'Please pass request body' });
  }
};

const deleteUser = async (req, res) => {
  if (req.body.id !== undefined) {
    const id = req.body.id;
    const deleteUser = await UserService.deleteUser(id);
    res.send(deleteUser);
  } else {
    res.send({ error: 'Please pass id in request body' });
  }
};

const auth = async (req, res) => {
  const { name, email, password, handedness } = req.body;

  // Look up the user entry in the database
  //   const user = db
  //     .get('users')
  //     .value()
  //     .filter((user) => email === user.email);

  const user = await UserService.getOneUser(email);

  // If found, compare the hashed passwords and generate the JWT token for the user
  if (user.length === 1) {
    bcrypt.compare(password, user[0].password, function (_err, result) {
      if (!result) {
        return res.status(401).json({ message: 'Invalid password' });
      } else {
        let loginData = {
          email,
          signInTime: Date.now()
        };

        const token = jwt.sign(loginData, jwtSecretKey);
        res.status(200).json({ message: 'success', token });
      }
    });
    // If no user is found, hash the given password and create a new entry in the auth db with the email and hashed password
  } else if (user.length === 0) {
    bcrypt.hash(password, 10, async function (_err, hash) {
      console.log({ name, email, password: hash, handedness });

      //db.get('users').push({ userId, name, email, password: hash, handedness }).write();
      await userService.addOneUser({ name, email, password: hash, handedness });

      let loginData = {
        email,
        signInTime: Date.now()
      };

      const token = jwt.sign(loginData, jwtSecretKey);
      res.status(200).json({ message: 'success', token });
    });
  }
};

const verify = async (req, res) => {
  const tokenHeaderKey = 'jwt-token';
  const authToken = req.headers[tokenHeaderKey];
  try {
    const verified = jwt.verify(authToken, jwtSecretKey);
    if (verified) {
      return res.status(200).json({ status: 'logged in', message: 'success' });
    } else {
      // Access Denied
      return res.status(401).json({ status: 'invalid auth', message: 'error' });
    }
  } catch (error) {
    // Access Denied
    return res.status(401).json({ status: 'invalid auth', message: 'error' });
  }
};

const checkAccount = async (req, res) => {
  console.log('here!');
  const email = req.body.email;

  console.log(req.body);

  //   const user = db
  //     .get('users')
  //     .value()
  //     .filter((user) => email === user.email);

  const user = await UserService.getOneUser(email);

  console.log(user);

  res.status(200).json({
    status: user.length === 1 ? 'User exists' : 'User does not exist',
    userExists: user.length === 1
  });
};

export default {
  getAllUsers,
  getOneUser,
  addNewUser,
  deleteUser,
  auth,
  verify,
  checkAccount
};
