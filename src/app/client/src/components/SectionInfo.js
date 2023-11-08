// src/SectionInfo.js

import React from 'react'
import '../index.css';

function SectionInfo() {
  return (
    <div className = "Section">
      <h1> Info </h1>
        <p>
          This application works by asking the user a prompt, 
          and the user can use the ASL they have learned to sign the 
          correct sign into their webcam. Then if they have signed correctly, 
          the application will let them know!
        </p>
    </div>
  )
}

export default SectionInfo