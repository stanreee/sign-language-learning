// import '../styles/Exercise.css';
import lettersResource from "../data/letters.json"
import React from "react";
 import YouTube from "react-youtube";
import a from "../images/learnLetters/a.png"
import b from "../images/learnLetters/b.png"
import c from "../images/learnLetters/c.png"
import d from "../images/learnLetters/d.png"
import e from "../images/learnLetters/e.png"
import f from "../images/learnLetters/f.png"
import g from "../images/learnLetters/g.png"
import h from "../images/learnLetters/h.png"
import i from "../images/learnLetters/i.png"
import j from "../images/learnLetters/j.png"
import k from "../images/learnLetters/k.png"
import l from "../images/learnLetters/l.png"
import m from "../images/learnLetters/m.png"
import n from "../images/learnLetters/n.png"
import o from "../images/learnLetters/o.png"
import p from "../images/learnLetters/p.png"
import q from "../images/learnLetters/q.png"
import r from "../images/learnLetters/r.png"
import s from "../images/learnLetters/s.png"
import t from "../images/learnLetters/t.png"
import u from "../images/learnLetters/u.png"
import v from "../images/learnLetters/v.png"
import w from "../images/learnLetters/w.png"
import x from "../images/learnLetters/x.png"
import y from "../images/learnLetters/y.png"
import z from "../images/learnLetters/z.png"

// https://www.youtube.com/watch?v=0vJ7tOVhGOw

const DisplayLetters = () => {

    //DATA CALL
    const allLetters = lettersResource.allLetters;
    const letterDesc = lettersResource.lettersDesc;

    return (
    <div className="Exercise-Page">
       <div className="Exercise-content">
        <h3>
            <img src={a} alt= "a" />
            <br></br>
            {allLetters[0]} 
            <br></br>
            {letterDesc[0]}
            <div className="Section"> </div>
            <img src={b} alt="b" />
            <br></br>
            {allLetters[1]} 
            <br></br>
            {letterDesc[1]}
            <div className="Section"> </div>
            <img src={c} alt="c" />
            <br></br>
            {allLetters[2]} 
            <br></br>
            {letterDesc[2]}
            <div className="Section"> </div>
            <img src={d} alt="d" />
            <br></br>
            {allLetters[3]} 
            <br></br>
            {letterDesc[3]}
            <div className="Section"> </div>
            <img src={e} alt="e" />
            <br></br>
            {allLetters[4]} 
            <br></br>
            {letterDesc[4]}
            <div className="Section"> </div>
            <img src={f} alt="f" />
            <br></br>
            {allLetters[5]} 
            <br></br>
            {letterDesc[5]}
            <div className="Section"> </div>
            <img src={g} alt="g" />
            <br></br>
            {allLetters[6]} 
            <br></br>
            {letterDesc[6]}
            <div className="Section"> </div>
            <img src={h} alt="h" />
            <br></br>
            {allLetters[7]} 
            <br></br>
            {letterDesc[7]}
            <div className="Section"> </div>
            <img src={i} alt="i" />
            <br></br>
            {allLetters[8]} 
            <br></br>
            {letterDesc[8]}
            <div className="Section"> </div>
            <img src={j} alt="j" />
            <br></br>
            {allLetters[9]} 
            <br></br>
            {letterDesc[9]}
            <div className="Section"> </div>
            <img src={k} alt="k" />
            <br></br>
            {allLetters[10]} 
            <br></br>
            {letterDesc[10]}
            <div className="Section"> </div>
            <img src={l} alt="l" />
            <br></br>
            {allLetters[11]} 
            <br></br>
            {letterDesc[11]}
            <div className="Section"> </div>
            <img src={m} alt="m" />
            <br></br>
            {allLetters[12]} 
            <br></br>
            {letterDesc[12]}
            <div className="Section"> </div>
            <img src={n} alt="n" />
            <br></br>
            {allLetters[13]} 
            <br></br>
            {letterDesc[13]}
            <div className="Section"> </div>
            <img src={o} alt="o" />
            <br></br>
            {allLetters[14]} 
            <br></br>
            {letterDesc[14]}
            <div className="Section"> </div>
            <img src={p} alt="p" />
            <br></br>
            {allLetters[15]} 
            <br></br>
            {letterDesc[15]}
            <div className="Section"> </div>
            <img src={q} alt="q" />
            <br></br>
            {allLetters[16]} 
            <br></br>
            {letterDesc[16]}
            <div className="Section"> </div>
            <img src={r} alt="r" />
            <br></br>
            {allLetters[17]} 
            <br></br>
            {letterDesc[17]}
            <div className="Section"> </div>
            <img src={s} alt="s" />
            <br></br>
            {allLetters[18]} 
            <br></br>
            {letterDesc[18]}
            <div className="Section"> </div>
            <img src={t} alt="t" />
            <br></br>
            {allLetters[19]} 
            <br></br>
            {letterDesc[19]}
            <div className="Section"> </div>
            <img src={u} alt="u" />
            <br></br>
            {allLetters[20]} 
            <br></br>
            {letterDesc[20]}
            <div className="Section"> </div>
            <img src={v} alt="v" />
            <br></br>
            {allLetters[21]} 
            <br></br>
            {letterDesc[21]}
            <div className="Section"> </div>
            <img src={w} alt="w" />
            <br></br>
            {allLetters[22]} 
            <br></br>
            {letterDesc[22]}
            <div className="Section"> </div>
            <img src={x} alt="x" />
            <br></br>
            {allLetters[23]} 
            <br></br>
            {letterDesc[23]}
            <div className="Section"> </div>
            <img src={y} alt="y" />
            <br></br>
            {allLetters[24]} 
            <br></br>
            {letterDesc[24]}
            <div className="Section"> </div>
            <img src={z} alt="z" />
            <br></br>
            {allLetters[25]} 
            <br></br>
            {letterDesc[25]}
        </h3>
      </div>
    </div>
    );
}

export default DisplayLetters;
