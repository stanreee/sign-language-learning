import React from 'react';
import { Routes, Route } from 'react-router-dom';
import Home from './pages/Home';
import Exercises from './pages/Exercises';
import Resources from './pages/Resources';
import NavBar from './components/NavBar';
import NoMatch from './components/NoMatch';


import './App.css';

const App = () => {
  return (
    <>
      <NavBar/>
       <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/Exercises" element={<Exercises />} />
          <Route path="/Resources" element={<Resources />} />
          <Route path="*" element={<NoMatch />} />
       </Routes>
    </>

  );
}

export default App;
