import React from 'react';
import { Routes, Route } from 'react-router-dom';
import Home from './pages/Home';
import Learn from './pages/Learn';
import Exercises from './pages/Exercises';
import Practice from './pages/Practice';
import NavBar from './components/NavBar';
import NoMatch from './components/NoMatch';
// import Quiz from './pages/Quiz';
import Account from './pages/Account';
import Login from './pages/Login';
import SignUp from './pages/SignUp';

import './App.css';

const App = () => {
  return (
    <>
      <NavBar/>
        <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/Learn" element={<Learn />} />
            <Route path="/Exercises" element={<Exercises />} />
            <Route path="/Practice" element={<Practice />} />
            <Route path="/Account" element={<Account />} />
            <Route path="/Login" element={<Login />} />
            <Route path="/SignUp" element={<SignUp />} />
            {/* <Route path="/Quiz" element={<Quiz />} /> */}
            <Route path="*" element={<NoMatch />} />
        </Routes>
    </>

  );
}

export default App;
