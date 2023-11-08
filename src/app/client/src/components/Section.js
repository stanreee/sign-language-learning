// src/Section.js

import React from 'react'
import './Section.css'

function Section(props) {
  return (
    <div className = "Section">
      <h1> {props.header1} </h1>
        <div className = "Section-Subtitle"> 
          {props.header2} {'\n'}
          {props.header3} {'\n \n'}
          {props.header4} 
        </div>
    </div>
  )
}

export default Section