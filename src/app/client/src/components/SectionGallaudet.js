// src/SectionGallaudet.js

import React from 'react'
import '../index.css';
import { Link } from 'react-router-dom';

function SectionGallaudet() {
  return (
    <div className = "Section">
      <h2> Gallaudet University </h2>
        <p>
            Gallaudet University is a university located in Washington, D.C.
            for the education of the deaf anf hard of hearing, which was 
            originally founded as a grammar school for both deaf anf blind
            children. They are striving to become the leading international 
            resource for research, innovation, and outreach related to deaf
            and hard-of-hearing people. To learn more about their cause and
            mission, click the link below!

            <div>
                <Link 
                  to="https://gallaudet.edu/about/#mission-vision#mission-vision"
                  style={{color: '#000',}}
                  >Go to Gallaudet University Website
                </Link>
            </div> 
        </p>
    </div>
  )
}

export default SectionGallaudet
