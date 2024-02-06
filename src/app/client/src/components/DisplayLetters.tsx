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
import youtube from "../images/learnLetters/youtube.png"

import lettersResource from "../data/letters.json"
import { Link } from "react-router-dom"

//DATA CALL
const allLetters = lettersResource.allLetters;
const letterDesc = lettersResource.lettersDesc;
const letterVideos = lettersResource.letterVideos;

const DisplayLetters = () => {
  return (
    <>
      <div className="container-letters">

        <div className = "card">
          <img src={a} alt= "a" />
          <h2>A</h2>
          {letterDesc[0]}
          <Link 
            to={`https://youtu.be/${letterVideos[0]}`}
            style={{color: '#000000',}}
            > <div > <img src={youtube} width={46.2} height={32.4} alt= "youtube" /> </div>
          </Link>
        </div>

        <div className = "card">
          <img src={b} alt= "b" />
          <h2>B</h2>
          {letterDesc[1]}
          <Link 
            to={`https://youtu.be/${letterVideos[1]}`}
            style={{color: '#000000',}}
            > <div > <img src={youtube} width={46.2} height={32.4} alt= "youtube" /> </div>
          </Link>
        </div>

        <div className = "card">
          <img src={c} alt= "c" />
          <h2>C </h2>
          {letterDesc[2]}
          <Link 
            to={`https://youtu.be/${letterVideos[2]}`}
            style={{color: '#000000',}}
            > <div > <img src={youtube} width={46.2} height={32.4} alt= "youtube" /> </div>
          </Link>
        </div>

        <div className = "card">
          <img src={d} alt= "d" />
          <h2>D </h2>
          {letterDesc[3]}
          <Link 
            to={`https://youtu.be/${letterVideos[3]}`}
            style={{color: '#000000',}}
            > <div > <img src={youtube} width={46.2} height={32.4} alt= "youtube" /> </div>
          </Link>
        </div>

        <div className = "card">
          <img src={e} alt= "e" />
          <h2>E </h2>
          {letterDesc[4]}
          <Link 
            to={`https://youtu.be/${letterVideos[4]}`}
            style={{color: '#000000',}}
            > <div > <img src={youtube} width={46.2} height={32.4} alt= "youtube" /> </div>
          </Link>
        </div>

        <div className = "card">
          <img src={f} alt= "f" />
          <h2>F </h2>
          {letterDesc[5]}
          <Link 
            to={`https://youtu.be/${letterVideos[5]}`}
            style={{color: '#000000',}}
            > <div > <img src={youtube} width={46.2} height={32.4} alt= "youtube" /> </div>
          </Link>
        </div>

        <div className = "card">
          <img src={g} alt= "g" />
          <h2>G </h2>
          {letterDesc[6]}
          <Link 
            to={`https://youtu.be/${letterVideos[6]}`}
            style={{color: '#000000',}}
            > <div > <img src={youtube} width={46.2} height={32.4} alt= "youtube" /> </div>
          </Link>
        </div>

        <div className = "card">
          <img src={h} alt= "h" />
          <h2>H </h2>
          {letterDesc[7]}
          <Link 
            to={`https://youtu.be/${letterVideos[7]}`}
            style={{color: '#000000',}}
            > <div > <img src={youtube} width={46.2} height={32.4} alt= "youtube" /> </div>
          </Link>
        </div>

        <div className = "card">
          <img src={i} alt= "i" />
          <h2>I </h2>
          {letterDesc[8]}
          <Link 
            to={`https://youtu.be/${letterVideos[8]}`}
            style={{color: '#000000',}}
            > <div > <img src={youtube} width={46.2} height={32.4} alt= "youtube" /> </div>
          </Link>
        </div>

        <div className = "card">
          <img src={j} alt= "j" />
          <h2>J </h2>
          {letterDesc[9]}
          <Link 
            to={`https://youtu.be/${letterVideos[9]}`}
            style={{color: '#000000',}}
            > <div > <img src={youtube} width={46.2} height={32.4} alt= "youtube" /> </div>
          </Link>
        </div>

        <div className = "card">
          <img src={k} alt= "k" />
          <h2>K </h2>
          {letterDesc[10]}
          <Link 
            to={`https://youtu.be/${letterVideos[10]}`}
            style={{color: '#000000',}}
            > <div > <img src={youtube} width={46.2} height={32.4} alt= "youtube" /> </div>
          </Link>
        </div>

        <div className = "card">
          <img src={l} alt= "l" />
          <h2>L </h2>
          {letterDesc[11]}
          <Link 
            to={`https://youtu.be/${letterVideos[11]}`}
            style={{color: '#000000',}}
            > <div > <img src={youtube} width={46.2} height={32.4} alt= "youtube" /> </div>
          </Link>
        </div>

        <div className = "card">
          <img src={m} alt= "m" />
          <h2>M </h2>
          {letterDesc[12]}
          <Link 
            to={`https://youtu.be/${letterVideos[12]}`}
            style={{color: '#000000',}}
            > <div > <img src={youtube} width={46.2} height={32.4} alt= "youtube" /> </div>
          </Link>
        </div>

        <div className = "card">
          <img src={n} alt= "n" />
          <h2>N </h2>
          {letterDesc[13]}
          <Link 
            to={`https://youtu.be/${letterVideos[13]}`}
            style={{color: '#000000',}}
            > <div > <img src={youtube} width={46.2} height={32.4} alt= "youtube" /> </div>
          </Link>
        </div>

        <div className = "card">
          <img src={o} alt= "o" />
          <h2>O </h2>
          {letterDesc[14]}
          <Link 
            to={`https://youtu.be/${letterVideos[14]}`}
            style={{color: '#000000',}}
            > <div > <img src={youtube} width={46.2} height={32.4} alt= "youtube" /> </div>
          </Link>
        </div>

        <div className = "card">
          <img src={p} alt= "p" />
          <h2>P </h2>
          {letterDesc[15]}
          <Link 
            to={`https://youtu.be/${letterVideos[15]}`}
            style={{color: '#000000',}}
            > <div > <img src={youtube} width={46.2} height={32.4} alt= "youtube" /> </div>
          </Link>
        </div>

        <div className = "card">
          <img src={q} alt= "q" />
          <h2>Q </h2>
          {letterDesc[16]}
          <Link 
            to={`https://youtu.be/${letterVideos[16]}`}
            style={{color: '#000000',}}
            > <div > <img src={youtube} width={46.2} height={32.4} alt= "youtube" /> </div>
          </Link>
        </div>

        <div className = "card">
          <img src={r} alt= "r" />
          <h2>R </h2>
          {letterDesc[17]}
          <Link 
            to={`https://youtu.be/${letterVideos[17]}`}
            style={{color: '#000000',}}
            > <div > <img src={youtube} width={46.2} height={32.4} alt= "youtube" /> </div>
          </Link>
        </div>

        <div className = "card">
          <img src={s} alt= "s" />
          <h2>S </h2>
          {letterDesc[18]}
          <Link 
            to={`https://youtu.be/${letterVideos[18]}`}
            style={{color: '#000000',}}
            > <div > <img src={youtube} width={46.2} height={32.4} alt= "youtube" /> </div>
          </Link>
        </div>

        <div className = "card">
          <img src={t} alt= "t" />
          <h2>T </h2>
          {letterDesc[19]}
          <Link 
            to={`https://youtu.be/${letterVideos[19]}`}
            style={{color: '#000000',}}
            > <div > <img src={youtube} width={46.2} height={32.4} alt= "youtube" /> </div>
          </Link>
        </div>

        <div className = "card">
          <img src={u} alt= "u" />
          <h2>U </h2>
          {letterDesc[20]}
          <Link 
            to={`https://youtu.be/${letterVideos[20]}`}
            style={{color: '#000000',}}
            > <div > <img src={youtube} width={46.2} height={32.4} alt= "youtube" /> </div>
          </Link>
        </div>

        <div className = "card">
          <img src={v} alt= "v" />
          <h2>V </h2>
          {letterDesc[21]}
          <Link 
            to={`https://youtu.be/${letterVideos[21]}`}
            style={{color: '#000000',}}
            > <div > <img src={youtube} width={46.2} height={32.4} alt= "youtube" /> </div>
          </Link>
        </div>

        <div className = "card">
          <img src={w} alt= "w" />
          <h2>W </h2>
          {letterDesc[22]}
          <Link 
            to={`https://youtu.be/${letterVideos[22]}`}
            style={{color: '#000000',}}
            > <div > <img src={youtube} width={46.2} height={32.4} alt= "youtube" /> </div>
          </Link>
        </div>

        <div className = "card">
          <img src={x} alt= "x" />
          <h2>X </h2>
          {letterDesc[23]}
          <Link 
            to={`https://youtu.be/${letterVideos[23]}`}
            style={{color: '#000000',}}
            > <div > <img src={youtube} width={46.2} height={32.4} alt= "youtube" /> </div>
          </Link>
        </div>

        <div className = "card">
          <img src={y} alt= "y" />
          <h2>Y </h2>
          {letterDesc[24]}
          <Link 
            to={`https://youtu.be/${letterVideos[24]}`}
            style={{color: '#000000',}}
            > <div > <img src={youtube} width={46.2} height={32.4} alt= "youtube" /> </div>
          </Link>
        </div>

        <div className = "card">
          <img src={z} alt= "z" />
          <h2>Z </h2>
          {letterDesc[25]}
          <Link 
            to={`https://youtu.be/${letterVideos[25]}`}
            style={{color: '#000000',}}
            > <div > <img src={youtube} width={46.2} height={32.4} alt= "youtube" /> </div>
          </Link>
        </div>
      </div>
    </>
  );
};

export default DisplayLetters;
