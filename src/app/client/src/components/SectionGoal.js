// src/SectionGoal.js

import React from 'react'
import '../index.css';
import { NavLink } from 'react-router-dom';

function SectionGoal() {
  return (
    <div className = "Section">
      <h1> Our Goal </h1>
        <p>
          Learning a new language can be an arduous task that only gets more challenging
          with age, as individuals may find it difficult to dedicate time and effort to
          it. American Sign Language (ASL) is particularly hard due to its visual and
          gestural nature, which is not found in other, verbal languages. ASLingo aims
          to ease that challenge by providing an online, easy-to-access web platform for
          individuals to learn new signs and test their comprehension at their own pace
          in a fun, interactive manner. With a focus on consistent effort and continuous
          feedback, ASLingo provides real-time guidance to ensure users stay on track to
          achieving their goals of learning ASL.
        </p>
        <NavLink 
						to="/exercises"
						style={({ isActive, isPending }) => {
							return {
							fontWeight: isActive ? "bold" : "bold",
							color: isPending ? "red" : '#fff',
							};
						}} >
						Click here to get started with your first lesson today!
				</NavLink>
    </div>
  )
}

export default SectionGoal