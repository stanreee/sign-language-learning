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
  const [email, setEmail] = useState('')

  useEffect(() => {
      // Fetch the user email and token from local storage
      const user = JSON.parse(localStorage.getItem('user'))
    
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
          setLoggedIn('success' === r.message)
          setEmail(user.email || '')
        })
    }, [])

  return (
    <>
      <NavBar/>
        <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/Learn" element={<Learn />} />
            <Route path="/Exercises" element={<Exercises />} />
            <Route path="/Practice" element={<Practice />} />
            <Route path="/Account" element={<Account email={email} loggedIn={loggedIn} setLoggedIn={setLoggedIn}/>} />
            <Route path="/Login" element={<Login setLoggedIn={setLoggedIn} setEmail={setEmail} />} />
            <Route path="/Signup" element={<SignUp setLoggedIn={setLoggedIn} setEmail={setEmail} />} />  
            {/* need to add handedness and name */}
            <Route path="/LoggedIn" element={<LoggedIn />} />
            <Route path="*" element={<NoMatch />} />
        </Routes>
    </>
  );
}

export default App;
