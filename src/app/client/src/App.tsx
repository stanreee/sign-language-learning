import React from 'react';
import { Routes, Route } from 'react-router-dom';
import Home from './pages/Home';
import Exercises from './pages/Exercises';
import Resources from './pages/Resources';
import NavBar from './components/NavBar';
import NoMatch from './components/NoMatch';
import Quiz from './pages/Quiz';


import './App.css';

const App = () => {
  return (
    <>
      <NavBar/>
       <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/Exercises" element={<Exercises />} />
          <Route path="/Resources" element={<Resources />} />
          {/* <Route path="/Quiz" element={<Quiz />} /> */}
          <Route path="*" element={<NoMatch />} />
       </Routes>
    </>

  );
}

export default App;
