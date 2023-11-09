// src/SectionHandSpeak.js

import React from 'react'
import '../index.css';
import { Link } from 'react-router-dom';

function SectionHandSpeak() {
  return (
    <div className = "Section">
      <h3> HandSpeak </h3>
        <p>
            HandSpeak is an online resource created by an ASL instructor
            and native signer in North Ameria. It includes an ASL video
            dictionary for learning signs, as well as many great videos
            to learn common signs or phrases that is great for beginners!
            Click the link below to learn more!

            <div>
                <Link 
                  to="https://www.handspeak.com/"
                  style={{color: '#000',}}
                  >Go to HandSpeak Website
                </Link>
            </div> 
        </p>
    </div>
  )
}

export default SectionHandSpeak
