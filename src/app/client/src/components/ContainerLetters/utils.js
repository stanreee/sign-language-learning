import React from "react";
import { Link } from "react-router-dom";
import wordsResource from "../../data/ASLSigns.json";
import youtube from "../../images/learnLetters/youtube.png"

export const wordCard = (num, index, width, height, first) => {
    // DATA CALL
    const allWords = wordsResource.words;
    const wordsDesc = wordsResource.wordsDesc;
    const wordVideos = wordsResource.wordVideos;

    return (
      <div className={first ? "card first" : "card"}>
        <img src={num} width={width} height={height} alt= {`${allWords[index]}`} />
        <h2>{allWords[index]}</h2>
        {wordsDesc[index]}
        <Link 
          to={`https://youtu.be/${wordVideos[index]}`}
          style={{color: '#000000',}}
          > <div > <img src={youtube} width={46.2} height={32.4} alt= "youtube" /> </div>
        </Link>
      </div>
    )
}