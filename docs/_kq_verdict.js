  // ---- verdict (inserted at /*__VERDICT__*/) ----
  const MATCH = D.suspect_matches || [];
  const matchTxt = MATCH.map((m) => { const r = m.row, e = r.el76;
    return `One simulated stage reproduced 2010 KQ's whole 1975 history: ${META[m.key].label} variant ${m.n} passed the Moon at ${fmt(r.moon_pass.r_km)} km on ${r.moon_pass.date.slice(0, 10)}, settled into a
    loose Earth orbit (first perigee ${r.first_perigee[0]}, ${fmt(r.first_perigee[1] / 1e3)} thousand km), came back to ${fmt(r.pass75[1] / 1e3)} thousand km on ${r.pass75[0]} and left the
    Earth on ${r.escape}, on an orbit around the Sun with a = ${fmt(e.a, 4)} AU, e = ${fmt(e.e, 4)} and i = ${fmt(e.i, 2)}&deg;, against 2010 KQ's ${fmt(KQ76.a[0], 4)} AU, ${fmt(KQ76.e[0], 4)}
    and ${fmt(KQ76.i[0], 2)}&deg; (Figure 7 overlays it on the clones). It shows that a ${META[m.key].label.split(" (")[0]} stage <i>can</i> become 2010 KQ; it is not by itself proof that it
    did, since any object leaving through the Sun&ndash;Earth L1 gateway in May 1975 lands on a similar orbit, and a single success in ${ST[m.key].n} runs is a small number.`; }).join(" ");
  const rate = (k) => ST[k] ? ST[k].esc75 / ST[k].n : 0;
  const r22 = rate("luna22"), r23 = rate("luna23"), r4 = rate("luna4");
  const cell = (cls, txt) => `<span class="${cls}">${txt}</span>`;
  const yes = (t) => cell("yes", t), no = (t) => cell("no", t), maybe = (t) => cell("maybe", t);
  const escTxt = (k) => ST[k] ? `${ST[k].esc75} of ${ST[k].n}` : "&ndash;";
  $("verdictTable").innerHTML = table(["Suspect", "Slow enough to become 2010 KQ? (§6)", "Clones that touch the Earth in its launch window (§7)",
    "Simulated stage leaves in May&ndash;June 1975 (§8)", "Size and sunlight push", "Verdict"], [
    ["Helios 1 Centaur (1974)", "&ndash;", `${SUP.helios1.n} <small>(and re-entered in 1979)</small>`, "&ndash;", "&ndash;", no("ruled out (re-entered 1979)")],
    ["Luna 1 Blok-E (1959)", no(`no: ${fmt(vinf(D.luna1_T), 1)} km/s vs &le; ${fmt(vinf(tisRange[0]), 1)}`), `${SUP.luna1.n}`, "&ndash;", maybe("plausible"), no("ruled out")],
    ["Luna 4 Blok-L (1963)", yes("yes"), no(`${SUP.luna4.n}`), r4 > 0 ? maybe(escTxt("luna4")) : no(escTxt("luna4")), yes("good fit"), no("very unlikely")],
    ["Luna 22 Blok-D (1974)", yes("yes"), yes(`${SUP.luna22.n}`), r22 > 0 ? yes(escTxt("luna22")) : maybe(escTxt("luna22")), maybe("stage looks 2&ndash;3&times; too big"), yes("possible")],
    ["Luna 23 Blok-D (1974)", yes("yes"), yes(`${SUP.luna23.n}`), r23 > 0 ? yes(escTxt("luna23")) : maybe(escTxt("luna23")), maybe("stage looks 2&ndash;3&times; too big"), yes("possible")]], []);
  // only claim a preference when both tests agree and the escape counts differ clearly (>= 5 apart, >= 3:1)
  const e22 = ST.luna22 ? ST.luna22.esc75 : 0, e23 = ST.luna23 ? ST.luna23.esc75 : 0;
  const strong = (a, b) => a - b >= 5 && a >= 3 * Math.max(b, 1);
  const m22 = ST.luna22 ? ST.luna22.match : 0, m23 = ST.luna23 ? ST.luna23.match : 0;
  const lean22 = r22 > r23 && SUP.luna22.n > SUP.luna23.n && m22 >= m23, lean23 = r23 > r22 && SUP.luna23.n > SUP.luna22.n && m23 >= m22;
  const which = lean22 && strong(e22, e23) ? "Luna 22 is the more likely of the two" : lean23 && strong(e23, e22) ? "Luna 23 is the more likely of the two" :
    lean22 ? "the orbit cannot yet tell which, though both tests lean slightly towards Luna 22" :
    lean23 ? "the orbit cannot yet tell which, though both tests lean slightly towards Luna 23" : "the orbit cannot yet tell which";
  $("verdictShort").innerHTML = `2010 KQ is most likely the Proton Blok-D upper stage of one of the two Soviet lunar launches of 1974, <b>Luna 22</b> (May 29) or <b>Luna 23</b>
    (October 28); ${which}. Luna 1 is ruled out and Luna 4 is very unlikely.`;
  $("verdictText").innerHTML = `Putting the pieces together: the 2010 data alone make 2010 KQ hollow hardware. Conservation of its slow relative speed rules out anything that left
  the Earth on an escape trajectory, including Luna 1's stage. The cloud of orbits consistent with the data says the object was looping around the Earth in 1974 and that the
  only orbits that can have started on the Earth after Sputnik do so in late 1973 or 1974, clustering in the weeks after the two Soviet lunar launches of that year and never
  near Luna 4's in 1963. Flying the suspects forward agrees: a Luna 4 stage that stays near the Earth until 1975 is uncommon (${pct(ST.luna4.bound75 / ST.luna4.n)} of runs), and none
  of ${ST.luna4.n} left in May&ndash;June 1975, whereas most 1974 Blok-D stages were still in Earth orbit in April 1975 (${pct(ST.luna22.bound75 / ST.luna22.n)} for Luna 22${ST.luna23 && ST.luna23.n ?
  `, ${pct(ST.luna23.bound75 / ST.luna23.n)} for Luna 23` : ""}) and some left at just the right time.<br><br>
  So the most likely answer is a Blok-D from <b>Luna 22 or Luna 23</b>, and ${which}: in the cloud ${SUP.luna22.n} vs ${SUP.luna23.n} clones point to their launch windows, and in the forward runs
  ${escTxt("luna22")} vs ${escTxt("luna23")} stages left with 2010 KQ's timing${MATCH.length ? `, but the only run that also lands on 2010 KQ's orbit is a ${META[MATCH[0].key].label.split(" (")[0]} stage` : ""}. Luna 22 was not on Miles's list; it was launched on the same rocket five months earlier and
  fits the orbit about as well. ${MATCH.length ? matchTxt : ''}<br><br>The one piece of evidence that points the other way is size: a Blok-D presents two to three times the area that 2010 KQ's brightness and sunlight push
  suggest, and Luna 4's Blok-L would fit them better. That tension is within the factor-of-two uncertainty of both estimates (reflectivity, tumbling), but it is the
  first thing a new observation should test.${MATCH.length ? "" : ` None of the simulated stages reproduced 2010 KQ's 1976 orbit; with a chaotic capture and a few hundred runs per suspect,
  nobody should expect one to.`} The case would be closed by recovering the object in late 2035, which would sharpen the 1975 pass, or by a spectrum or light curve compared
  with a known Blok-D.`;
