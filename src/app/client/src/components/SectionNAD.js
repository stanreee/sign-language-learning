// src/SectionNAD.js

import React from 'react'
import '../index.css';
import { Link } from 'react-router-dom';

function SectionNAD() {
  return (
    <div className = "Section">
      <h2> National Association of the Deaf </h2>
        <p>
            The National Association of the Deaf (NAD) is an organization for 
            the promotion of the right of deaf people in the United States and 
            is a non-profit organization run by Deaf people to advocate for 
            deaf rights. They have great resources to check out, here are there 
            FAQ's about the Deaf community and culture.

            <div>
                <Link 
                  to="https://www.nad.org/resources/american-sign-language/community-and-culture-frequently-asked-questions/"
                  style={{color: '#000',}}
                  >Go to NAD Website
                </Link>
            </div> 
        </p>
    </div>
  )
}

export default SectionNAD
