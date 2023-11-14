// src/SectionResources.js

import React from 'react'
import '../index.css';
import { NavLink } from 'react-router-dom';

function SectionResources() {
  return (
    <div className = "Section">
      <h1> Additional Resources </h1>
        <p>
          The goal of ASLingo is to allow people all of all different abilities and skill levels learn ASL, American Sign Language.
          There are many links to check out if you would like to learn more about the culture and community of Deaf or hard of hearing people.
        </p>

        <NavLink 
						to="/resources"
						style={({ isActive, isPending }) => {
							return {
							fontWeight: isActive ? "bold" : "bold",
							color: isPending ? "red" : '#fff',
							};
						}} >
						Click here to go to the Resources page to learn more!
				</NavLink>
    </div>
  )
}

export default SectionResources