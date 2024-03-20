import React from 'react';
import { Routes, Route } from 'react-router-dom';
import Home from './pages/Home';
import Learn from './pages/Learn';
import Exercises from './pages/Exercises';
import Practice from './pages/Practice';
import NavBar from './components/NavBar';
import NoMatch from './components/NoMatch';
import Account from './pages/Account';
import Login from './pages/Login';
import SignUp from './pages/SignUp';
import LoggedIn from './pages/LoggedIn'

import { useEffect, useState } from 'react'

import './App.css';

const App = () => {

  const [loggedIn, setLoggedIn] = useState(false)
  const [name, setName] = useState('')
  const [email, setEmail] = useState('')
  const [userId, setUserId] = useState('')

  useEffect(() => {
      // Fetch the user email and token from local storage
      const user = JSON.parse(localStorage.getItem('user'))
    
      console.log(user)
      // If the token/email does not exist, mark the user as logged out
      if (!user || !user.token) {
        setLoggedIn(false)
        return
      }
    
      // If the token exists, verify it with the auth server to see if it is valid
      fetch('http://localhost:3080/verify', {
        method: 'POST',
        headers: {
          'jwt-token': user.token,
        },
      })
        .then((r) => r.json())
        .then((r) => {
          console.log('success sign on')
          setLoggedIn('success' === r.message)
          console.log(user.email)
          setEmail(user.email)
          console.log(email)
        })
          

        // fetch('http://localhost:3080/get-user-email', {
        //     method: 'POST',
        //     headers: {
        //       'Content-Type': 'application/json',
        //     },
        //     body: JSON.stringify({ email })
        //   })
        //   .then((r) => {
        //     setName(r.name || '')
        //     setUserId(r.userId || '')
        //   })
        
        
    }, [])

  return (
    <>
      <NavBar/>
        <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/Learn" element={<Learn />} />
            <Route path="/Exercises" element={<Exercises userEmail={email}/>} />
            <Route path="/Practice" element={<Practice />} />
            <Route path="/Account" element={<Account setLoggedIn={setLoggedIn} userId={userId} email={email} name={name} loggedIn={loggedIn} />} />
            <Route path="/Login" element={<Login setLoggedIn={setLoggedIn} setName={setName} setEmail={setEmail} />} />
            <Route path="/Signup" element={<SignUp setLoggedIn={setLoggedIn} setName={setName} setEmail={setEmail} />} />  
            <Route path="/LoggedIn" element={<LoggedIn />} />
            <Route path="*" element={<NoMatch />} />
        </Routes>
    </>
  );
}

export default App;
