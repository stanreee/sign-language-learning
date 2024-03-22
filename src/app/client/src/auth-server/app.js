//DEPRECTAED

// const express = require('express')
// const bcrypt = require('bcrypt')
// var cors = require('cors')
// const jwt = require('jsonwebtoken')
// var low = require('lowdb')
// var FileSync = require('lowdb/adapters/FileSync')
// var adapter = new FileSync('./users.json')
// var db = low(adapter)

// // Initialize Express app
// const app = express()

// // Define a JWT secret key. This should be isolated by using env variables for security
// const jwtSecretKey = 'dsfdsfsdfdsvcsvdfgefg'

// // Set up CORS and JSON middlewares
// app.use(cors())
// app.use(express.json())
// app.use(express.urlencoded({ extended: true }))

// // Basic home route for the API
// app.get('/', (_req, res) => {
//     res.send('Auth API.\nPlease use POST /auth & POST /verify for authentication')
//   })

// // The auth endpoint that creates a new user record or logs a user based on an existing record
// app.post('/auth', (req, res) => {
//     const { userId, name, email, password, handedness } = req.body
  
//     // Look up the user entry in the database
//     const user = db
//       .get('users')
//       .value()
//       .filter((user) => email === user.email)
  
//     // If found, compare the hashed passwords and generate the JWT token for the user
//     if (user.length === 1) {
//       bcrypt.compare(password, user[0].password, function (_err, result) {
//         if (!result) {
//           return res.status(401).json({ message: 'Invalid password' })
//         } else {
//           let loginData = {
//             userId,
//             email,
//             signInTime: Date.now(),
//           }
  
//           const token = jwt.sign(loginData, jwtSecretKey)
//           res.status(200).json({ message: 'success', token })
//         }
//       })
//       // If no user is found, hash the given password and create a new entry in the auth db with the email and hashed password
//     } else if (user.length === 0) {
//       bcrypt.hash(password, 10, function (_err, hash) {
//         console.log({ userId, name, email, password: hash, handedness })
//         db.get('users').push({ userId, name, email, password: hash, handedness }).write()
  
//         let loginData = {
//           email,
//           signInTime: Date.now(),
//         }
  
//         const token = jwt.sign(loginData, jwtSecretKey)
//         res.status(200).json({ message: 'success', token })
//       })
//     }
//   })

//   // The verify endpoint that checks if a given JWT token is valid
// app.post('/verify', (req, res) => {
//     const tokenHeaderKey = 'jwt-token'
//     const authToken = req.headers[tokenHeaderKey]
//     try {
//       const verified = jwt.verify(authToken, jwtSecretKey)
//       if (verified) {
//         return res.status(200).json({ status: 'logged in', message: 'success' })
//       } else {
//         // Access Denied
//         return res.status(401).json({ status: 'invalid auth', message: 'error' })
//       }
//     } catch (error) {
//       // Access Denied
//       return res.status(401).json({ status: 'invalid auth', message: 'error' })
//     }
//   })

//   // An endpoint to see if there's an existing account for a given email address
// app.post('/check-account', (req, res) => {
//     const { email } = req.body
  
//     console.log(req.body)
  
//     const user = db
//       .get('users')
//       .value()
//       .filter((user) => email === user.email)
  
//     console.log(user)
  
//     res.status(200).json({
//       status: user.length === 1 ? 'User exists' : 'User does not exist',
//       userExists: user.length === 1,
//     })
//   })

// app.post('/get-stats', (req, res) =>{
//     const { userId } = req.body
//     console.log('here')
//     console.log(req.body)
  
//     const user = db
//       .get('users')
//       .value()
//       .filter((user) => userId === user.userId)
  
//     console.log(user)
  
//     res.status(200).json(user)
//   })

//   app.post('/post-quiz', (req, res) =>{
//     const { results, userEmail } = req.body
//     console.log('post quiz')
//     console.log(req.body)

//     const resultArr = []
  
//     const skill = db
//       .get('skill')
//       .value()
//       .filter((user) => userEmail.userEmail === user.email)


  
//     console.log(skill)
//     console.log(skill.questionsAtt)

//     console.log(results.length)
//     for(let i = 0; i < results.length; i++){
//       skill.questionsAtt += 1
//       //if(results[i].isCorrect) {
//         console.log(results[i].isCorrect)
//         skill.questionsCorr += 1
//       //} 
//     }

//     console.log('post')
//     console.log(skill)

//     db.get('users').push(skill).write()
  
//     res.status(200).json(skill)
//   })

//   app.post('/get-stats', (req, res) =>{
//     const { userEmail } = req.body
//     console.log(req.body)
  
//     const user = db
//       .get('users')
//       .value()
//       .filter((user) => userEmail === user.email)
  
//     console.log(user)
  
//     res.status(200).json(user)
//   })

//   app.post('/get-user-email', (req, res) =>{
//     const { email } = req.body
//     console.log('get user email')
//     console.log(req.body)
  
//     const user = db
//       .get('users')
//       .value()
//       .filter((user) => email === user.email)

//     console.log(user)
    
//     res.status(200).json(user.name, user.userId)
//   })
  

//   app.listen(3080)