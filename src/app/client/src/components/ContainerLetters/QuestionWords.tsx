import React from "react";

import "./ContainerLetters.css";
import { wordCard } from "./utils";

import who from "../../images/learnWords/who.png"
import what from "../../images/learnWords/what.png"
import where from "../../images/learnWords/where.png"
import when from "../../images/learnWords/when.png"
import why from "../../images/learnWords/why.png"
import how from "../../images/learnWords/how.png"

const QuestionWords = (props: any) => {
    const {
        className
    } = props;

    return (
        <div style={{overflow: "hidden", width: "75%"}} className={className}>
            <div className="container-letters">
                <div id="who" data-hs-anchor="true">{wordCard(who, 0, 370, 270, true)}</div>
                <div id="what" data-hs-anchor="true">{wordCard(what, 1, 270, 270)}</div>
                <div id="where" data-hs-anchor="true">{wordCard(where, 2, 370, 270)}</div>
                <div id="when" data-hs-anchor="true">{wordCard(when, 3, 370, 270)}</div>
                <div id="why" data-hs-anchor="true">{wordCard(why, 4, 370, 270)}</div>
                <div id="how" data-hs-anchor="true">{wordCard(how, 5, 670, 270)}</div>       
            </div>

            <div className="jump-to">
                Jump to:
                <br />
                <a href="#who" rel="noopener">Who?   </a>
                <a href="#what" rel="noopener">What?   </a>
                <a href="#where" rel="noopener">Where?   </a>
                <a href="#when" rel="noopener">When?   </a>
                <a href="#why" rel="noopener">Why?   </a>
                <a href="#how" rel="noopener">How?   </a>
            </div>
        </div>
    );
}

export default QuestionWords;