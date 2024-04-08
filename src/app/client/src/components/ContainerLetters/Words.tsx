import React from "react";

import hello from "../../images/learnWords/hello.png"
import please from "../../images/learnWords/please.png"
import thanku from "../../images/learnWords/thanku.png"
import yes from "../../images/learnWords/yes.png"
import no from "../../images/learnWords/no.png"
import need from "../../images/learnWords/need.png"
import home from "../../images/learnWords/home.png"
import family from "../../images/learnWords/family.png"
import friend from "../../images/learnWords/friend.png"
import future from "../../images/learnWords/future.png"
import spaghetti from "../../images/learnWords/spaghetti.png"

import "./ContainerLetters.css";

import { wordCard } from "./utils";

const Words = (props: any) => {
    const {
        className
    } = props;

    return (
        <div style={{overflow: "hidden", width: "75%"}} className={className}>
            <div className="container-letters">
                <div id="hello" data-hs-anchor="true">{wordCard(hello, 6, 370, 270, true)}</div>               
                <div id="please" data-hs-anchor="true">{wordCard(please, 7, 270, 270)}</div>      
                <div id="thanku" data-hs-anchor="true">{wordCard(thanku, 8, 570, 270)}</div>      
                <div id="yes" data-hs-anchor="true">{wordCard(yes, 9, 300, 270)}</div>         
                <div id="no" data-hs-anchor="true">{wordCard(no, 10, 570, 270)}</div>       
                <div id="need" data-hs-anchor="true">{wordCard(need, 11, 570, 270)}</div>       
                <div id="home" data-hs-anchor="true">{wordCard(home, 12, 670, 270)}</div>  
                <div id="family" data-hs-anchor="true">{wordCard(family, 13, 370, 270)}</div> 
                <div id="friend" data-hs-anchor="true">{wordCard(friend, 14, 300, 300)}</div> 
                <div id="future" data-hs-anchor="true">{wordCard(future, 15, 670, 270)}</div> 
                <div id="spaghetti" data-hs-anchor="true">{wordCard(spaghetti, 16, 370, 270)}</div> 
            </div>

            <div className="jump-to">
                Jump to:
                <br />
                <a href="#hello" rel="noopener">Hello,   </a>
                <a href="#please" rel="noopener">Please,  </a>
                <a href="#thanku" rel="noopener">Thank You   </a>
                <br />
                <a href="#yes" rel="noopener">Yes,   </a>
                <a href="#no" rel="noopener">No,   </a>
                <a href="#need" rel="noopener">Need,   </a>
                <a href="#home" rel="noopener">Home   </a>
                <br />
                <a href="#family" rel="noopener">Family,   </a>
                <a href="#friend" rel="noopener">Friend,   </a>
                <a href="#future" rel="noopener">Future,   </a>
                <a href="#spaghetti" rel="noopener">Spaghetti   </a>
            </div>
        </div>
    );
}

export default Words;