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

import lettersResource from "../data/letters.json"
import VideoClip from "./VideoClip"

//DATA CALL
const allLetters = lettersResource.allLetters;
const letterDesc = lettersResource.lettersDesc;
const letterVideos = lettersResource.letterVideos;

const DisplayLetters = () => {
  return (
    <>
      <div className="container-letters">

        <div className = "card">
          <div className = "letterAlign">
            <img src={a} alt= "a" /> 
            <VideoClip letterVideoId= {letterVideos[0]}/>
          </div> 
          <h2>{allLetters[0]} </h2>
          {letterDesc[0]}
        </div>

        <div className = "card">
          <div className = "letterAlign">
            <img src={b} alt= "b" /> 
            <VideoClip letterVideoId= {letterVideos[1]}/>
          </div> 
          <h2>{allLetters[1]} </h2>
          {letterDesc[1]}
        </div>

        <div className="card"> 
          <div className = "letterAlign">
            <img src={c} alt="c" />
            <VideoClip letterVideoId= {letterVideos[2]}/>
          </div>
          <h2>{allLetters[2]} </h2>
          {letterDesc[2]}
        </div>

        <div className="card"> 
          <div className = "letterAlign">
            <img src={d} alt="d" />
            <VideoClip letterVideoId= {letterVideos[3]}/>
          </div>
          <h2>{allLetters[3]} </h2>
          {letterDesc[3]}
        </div>

        <div className="card"> 
          <div className = "letterAlign">
            <img src={e} alt="e" />
            <VideoClip letterVideoId= {letterVideos[4]}/>
          </div>
          <h2>{allLetters[4]} </h2>
          {letterDesc[4]}
        </div>

        <div className="card"> 
          <div className = "letterAlign">
            <img src={f} alt="f" />
            <VideoClip letterVideoId= {letterVideos[5]}/>
          </div>
          <h2>{allLetters[5]} </h2>
          {letterDesc[5]}
        </div>

        <div className="card"> 
          <div className = "letterAlign">
            <img src={g} alt="g" />            
            <VideoClip letterVideoId= {letterVideos[6]}/>
          </div>
          <h2>{allLetters[6]} </h2>
          {letterDesc[6]}
        </div>

        <div className="card"> 
          <div className = "letterAlign">
            <img src={h} alt="h" />            
            <VideoClip letterVideoId= {letterVideos[7]}/>
          </div>
          <h2>{allLetters[7]} </h2>
          {letterDesc[7]}
        </div>

        <div className="card"> 
          <div className = "letterAlign">
            <img src={i} alt="i" />            
            <VideoClip letterVideoId= {letterVideos[8]}/>
          </div>
          <h2>{allLetters[8]} </h2>
          {letterDesc[8]}
        </div>

        <div className="card"> 
          <div className = "letterAlign">
            <img src={j} alt="j" />            
            <VideoClip letterVideoId= {letterVideos[9]}/>
          </div>
          <h2>{allLetters[9]} </h2>
          {letterDesc[9]}
        </div>

        <div className="card"> 
          <div className = "letterAlign">
            <img src={k} alt="k" />            
            <VideoClip letterVideoId= {letterVideos[10]}/>
          </div>
          <h2>{allLetters[10]} </h2>
          {letterDesc[10]}
        </div>

        <div className="card"> 
          <div className = "letterAlign">
            <img src={l} alt="l" />           
            <VideoClip letterVideoId= {letterVideos[11]}/>
          </div>
           <h2>{allLetters[11]} </h2>
          {letterDesc[11]}
        </div>

        <div className="card"> 
          <div className = "letterAlign">
            <img src={m} alt="m" />            
            <VideoClip letterVideoId= {letterVideos[12]}/>
          </div>
          <h2>{allLetters[12]} </h2>
          {letterDesc[12]}
        </div>

        <div className="card"> 
          <div className = "letterAlign">
            <img src={n} alt="n" />            
            <VideoClip letterVideoId= {letterVideos[13]}/>
          </div>
          <h2>{allLetters[13]} </h2>
          {letterDesc[13]}
        </div>

        <div className="card"> 
          <div className = "letterAlign">
            <img src={o} alt="o" />            
            <VideoClip letterVideoId= {letterVideos[14]}/>
          </div>
          <h2>{allLetters[14]} </h2>
          {letterDesc[14]}
        </div>

        <div className="card"> 
          <div className = "letterAlign">
            <img src={p} alt="p" />            
            <VideoClip letterVideoId= {letterVideos[15]}/>
          </div>
          <h2>{allLetters[15]} </h2>
          {letterDesc[15]}
        </div>

        <div className="card"> 
          <div className = "letterAlign">
            <img src={q} alt="q" />            
            <VideoClip letterVideoId= {letterVideos[16]}/>
          </div>
          <h2>{allLetters[16]} </h2>
          {letterDesc[16]}
        </div>

        <div className="card"> 
          <div className = "letterAlign">
            <img src={r} alt="r" />            
            <VideoClip letterVideoId= {letterVideos[17]}/>
          </div>
          <h2>{allLetters[17]} </h2>
          {letterDesc[17]}
        </div>

        <div className="card"> 
          <div className = "letterAlign">
            <img src={s} alt="s" />            
            <VideoClip letterVideoId= {letterVideos[18]}/>
          </div>
          <h2>{allLetters[18]} </h2>
          {letterDesc[18]}
        </div>

        <div className="card"> 
          <div className = "letterAlign">
            <img src={t} alt="t" />            
            <VideoClip letterVideoId= {letterVideos[19]}/>
          </div>
          <h2>{allLetters[19]} </h2>
          {letterDesc[19]}
        </div>

        <div className="card"> 
          <div className = "letterAlign">
            <img src={u} alt="u" />            
            <VideoClip letterVideoId= {letterVideos[20]}/>
          </div>
          <h2>{allLetters[20]} </h2>
          {letterDesc[20]}
        </div>

        <div className="card"> 
          <div className = "letterAlign">
            <img src={v} alt="v" />            
            <VideoClip letterVideoId= {letterVideos[21]}/>
          </div>
          <h2>{allLetters[21]} </h2>
          {letterDesc[21]}
        </div>

        <div className="card"> 
          <div className = "letterAlign">
            <img src={w} alt="w" />            
            <VideoClip letterVideoId= {letterVideos[22]}/>
          </div>
          <h2>{allLetters[22]} </h2>
          {letterDesc[22]}
        </div>

        <div className="card"> 
          <div className = "letterAlign">
            <img src={x} alt="x" />            
            <VideoClip letterVideoId= {letterVideos[23]}/>
          </div>
          <h2>{allLetters[23]} </h2>
          {letterDesc[23]}
        </div>

        <div className="card"> 
          <div className = "letterAlign">
            <img src={y} alt="y" />            
            <VideoClip letterVideoId= {letterVideos[24]}/>
          </div>
          <h2>{allLetters[24]} </h2>
          {letterDesc[24]}
        </div>

        <div className="card"> 
          <div className = "letterAlign">
            <img src={z} alt="z" />            
            <VideoClip letterVideoId= {letterVideos[25]}/>
          </div>
          <h2>{allLetters[25]} </h2>
          {letterDesc[25]}
        </div>
        
      </div>
      
    </>
  );
};

export default DisplayLetters;
