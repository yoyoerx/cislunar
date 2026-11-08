// ---- texts (inserted into the template at /*__TEXTS__*/ by build_case_study_kq.py) ----
const SUP = D.back.support;
const susStats = (k) => {
  const r = SUS[k], n = r.length;
  const fp = r.filter((x) => x.first_perigee).map((x) => x.first_perigee[0]).sort();
  return {n, imp: r.filter((x) => x.impact).length, earlyEsc: r.filter((x) => x.escape && x.escape < "1975-01-01").length,
    bound75: r.filter((x) => x.bound_apr75).length, esc75: r.filter((x) => x.escape75).length, like: r.filter((x) => x.orbit_like).length,
    match: r.filter((x) => x.match).length, fp0: fp[Math.floor(fp.length * 0.1)], fp1: fp[Math.floor(fp.length * 0.9)]};
};
const ST = Object.fromEntries(SUSKEYS.map((k) => [k, susStats(k)]));
const FW = D.forward, FWimp = FW.filter((r) => r.impact).length;
const fwW = (() => { const w = Object.fromEntries(ROWS.map((r) => [r.i, r.weight_gray2])); let a = 0, b = 0;
  FW.forEach((r) => { a += (w[r.i] || 0) * (r.impact ? 1 : 0); b += (w[r.i] || 0); }); return a / b; })();
const nSolar = count((r) => r.cat === "solar"), nTouch = SUP.any.n;
const in22 = SUP.luna22.n, in23 = SUP.luna23.n;
const tisRange = [Math.min(...D.tisserand.T), Math.max(...D.tisserand.T)];
const vinf = (T) => 29.78 * Math.sqrt(Math.max(3 - T, 0));

(function texts() {
  $("headline").innerHTML = `<ul>
  <li><b>The 2010 data on their own say "rocket stage":</b> sunlight pushes on 2010 KQ with <i>C<sub>r</sub>A/m</i> = ${fmt(AMR, 4)} &plusmn; ${fmt(AMRS, 4)} m&sup2;/kg,
  ${fmt(AMR / AMRS, 0)} standard deviations from zero and a hundred times what a rock would feel (Gray found ${G1.amr} &plusmn; ${G1.amr_sigma} from the same data).</li>
  <li><b>Luna 1 is ruled out.</b> Its stage left the Earth at about ${fmt(vinf(D.luna1_T), 1)} km/s; 2010 KQ has never been faster than ${fmt(vinf(tisRange[0]), 1)} km/s relative to us,
  and close passes cannot bridge that gap (section 6).</li>
  <li><b>2010 KQ was in orbit around the Earth in 1974.</b> Of ${N} orbits that fit the 2010 data, ${nTouch} could have started on the Earth after 1957, and
  ${count((r) => r.touch_date && r.touch_date >= "1973-11-01" && r.touch_date < "1975-01-01")} of those touch it between November 1973 and December 1974; none goes near Luna 1's 1959
  or Luna 4's 1963 launch. ${in22 + in23} touch it in the weeks just after the two Soviet lunar launches of 1974, Luna 22 (${in22}) and Luna 23 (${in23}).</li>
  <li><b>Flying the suspects forward:</b> a simulated Luna 4 stage escaped in May&ndash;June 1975, as 2010 KQ did, in ${ST.luna4 ? ST.luna4.esc75 : "?"} of ${ST.luna4 ? ST.luna4.n : "?"} runs;
  Luna 22's in ${ST.luna22 ? ST.luna22.esc75 : "?"} of ${ST.luna22 ? ST.luna22.n : "?"}, Luna 23's in ${ST.luna23 ? ST.luna23.esc75 : "?"} of ${ST.luna23 ? ST.luna23.n : "?"}${
  (D.suspect_matches || []).length ? `; one Luna 23 stage reproduced 2010 KQ's April 1975 pass, its May 1975 escape and its later orbit around the Sun` : ""}.</li>
  <li><b>Verdict:</b> <span id="verdictShort"></span></li>
  <li><b>It comes back in January 2036</b> in every one of ${FW.length} clones run forward, and is captured again; ${FWimp} of them (${pct(FWimp / FW.length)}) hit the Earth in
  April 2036. That share is very sensitive to the sunlight push and should not be read as a forecast; recovering the object in late 2035 would settle both its fate and
  its identity.</li></ul>`;

  $("suspectTable").innerHTML = table(["Suspect", "Launched", "Stage", "Status before this study"], [
    ["Helios 1 Centaur (1974-097C)", "1974-12-10", "Centaur TC-2", "ruled out: its extended mission left it in Earth orbit, re-entered 1979 (J. Hopkins)"],
    ["Luna 1", "1959-01-02", "Blok-E, 1,472 kg", "open (heliocentric since 1959)"],
    ["Luna 4", "1963-04-02", "Blok-L, 1,150 kg empty, 2.7 m &times; 2.7 m", "open (missed the Moon by 8,500 km into an 89,250 &times; 694,000 km Earth orbit)"],
    ["Luna 23", "1974-10-28", "Blok-D, ~2.1&ndash;2.5 t empty, 3.7 m &times; ~6 m", "open; Wikipedia's favourite"],
    ["Luna 22 <small>(added here)</small>", "1974-05-29", "Blok-D (same vehicle as Luna 23)", "not considered by Miles"]], []);

  const F = D.fit;
  $("dataText").innerHTML = `The Minor Planet Center no longer serves 2010 KQ's observations: deleting an "asteroid" removes it from the database. They survive in Bill Gray's
  pseudo-MPEC, in the MPC's 80-column format: ${F.n_obs} positions (after removing two exact duplicates) from ${F.stations.length} observatories between ${F.first_utc} and
  ${F.last_utc}, a ${fmt(F.arc_days, 0)}-day arc. Miles's later observations (June 26 and July 6), which first showed the non-gravitational motion clearly, were never
  published, so this study has 5 weeks of data where Gray's final solution had 7.`;

  const p0 = D.profile.find((p) => p.amr === 0), c0 = Math.min(...D.profile.map((p) => p.chi2));
  $("fitText").innerHTML = `A first orbit from Gauss's method on the first ${fmt(F.iod.days, 1)} days (${F.iod.n} positions; ${fmt(F.iod.r_km / 1e3)} thousand km away) is grown
  over the whole arc in the full model (Earth, Moon, Sun, Earth's J2). With gravity alone the fit leaves an RMS residual of ${fmt(F.steps_gravity.at(-1).rms, 2)}&Prime;;
  letting sunlight push on the object brings it to ${fmt(D.rms, 2)}&Prime; with all ${F.n_used} positions kept. The improvement is worth Δχ&sup2; = ${fmt(p0.chi2 - c0)}
  &mdash; about ${fmt(Math.sqrt(p0.chi2 - c0), 0)}&sigma;. A 10 m rock would show <i>C<sub>r</sub>A/m</i> &asymp; 0.0001; an empty upper stage 0.005&ndash;0.02.`;
  $("fitTable").innerHTML = table(["Quantity", "This study", "Gray, June 2010 (same data)", "Gray, July 2010 (+ Miles)"], [
    ["Observations", `${F.n_used} of ${F.n_obs}`, `${G1.n_obs}`, "+ June 26, July 6"],
    ["RMS residual", `${fmt(D.rms, 2)}&Prime;`, `${G1.rms_arcsec}&Prime;`, "&ndash;"],
    ["<i>C<sub>r</sub>A/m</i> [m&sup2;/kg]", `${fmt(AMR, 4)} &plusmn; ${fmt(AMRS, 4)}`, `${G1.amr} &plusmn; ${G1.amr_sigma}`, `${G2.amr} &plusmn; ${G2.amr_sigma}`],
    ["1975 closest approach", `${median(ROWS.filter((r) => r.enc_date).map((r) => r.enc_date))} (median), ${fmt(median(ROWS.filter((r) => r.enc_r_km).map((r) => r.enc_r_km)) / 1e3)} thousand km`,
     `${G1.encounter_1975.date} &plusmn; ${G1.encounter_1975.date_sigma_days} d, ${fmt(G1.encounter_1975.r_km / 1e3)} &plusmn; ${fmt(G1.encounter_1975.r_sigma_km / 1e3)} thousand km`,
     `${G2.encounter_1975.date} &plusmn; ${G2.encounter_1975.date_sigma_days} d, ${fmt(G2.encounter_1975.r_km / 1e3)} &plusmn; ${fmt(G2.encounter_1975.r_sigma_km / 1e3)} thousand km`]], []);

  const v = (lab, gr) => D.validation.find((x) => x.label === lab && x.gr === gr);
  const at = (o, d) => o.err_km[o.date.findIndex((x) => x <= d)];  // dates run backwards from 2010
  $("validText").innerHTML = `The first attempt failed this test badly: Cruithne drifted ${fmt(at(v("3753 Cruithne", false), "1992-07-01") / 1e3)} thousand km from JPL's
  position over 18 years. The cause was subtle. The toolkit integrates in a frame centred on the Earth, which accelerates with the Earth; relativity adds a tiny
  outward pull on the Earth (about 1.8&times;10<sup>-13</sup> km/s&sup2;) as well as on the object, and leaving the Earth's share out acts like a small force rotating once a
  year &mdash; in resonance with any object on a one-year orbit, like these two and 2010 KQ. With both terms in, Cruithne stays within ${fmt(at(v("3753 Cruithne", true), "1992-07-01"))}
  km of JPL after 18 years and 1991 VG within ${fmt(at(v("1991 VG", true), "1992-07-01"))} km up to its 1991 pass. That is far smaller than the uncertainty of 2010 KQ's own orbit.`;

  const ok = ROWS.filter((r) => r.enc_date), bnd = ok.filter((r) => r.enc_bound).length;
  $("encText").innerHTML = `Every clone passes close to the Earth in April 1975 (median ${median(ok.map((r) => r.enc_date))}, ${fmt(median(ok.map((r) => r.enc_r_km)) / 1e3)} thousand km), and in
  ${bnd} of ${ok.length} the object is gravitationally bound to the Earth at closest approach: in 1975 it was briefly a satellite of the Earth. Then it leaves for good in
  May 1975 (${ESC.length} clones: ${ESC[Math.floor(ESC.length * 0.05)]} to ${ESC[Math.floor(ESC.length * 0.95)]}). The date and distance depend mostly on the sunlight push. Clones with
  <i>C<sub>r</sub>A/m</i> near 0.0085&ndash;0.01 reproduce Gray's July solution, which used more data than is public, and those are the clones that had been looping around
  the Earth for months.`;

  $("luna1Text").innerHTML = `Luna 1 left the Earth on 1959 January 2 on a direct escape trajectory, and its 1,472 kg Blok-E stage went with it into an orbit of 0.978 &times; 1.318 AU
  around the Sun. For a body crossing the Earth's orbit, the <i>Tisserand parameter</i> combines semi-major axis, eccentricity and inclination into one number that
  close passes by the Earth leave almost unchanged; it is also a speedometer, since the speed relative to the Earth far from it is roughly 29.8&radic;(3&minus;T) km/s.
  2010 KQ's value stays between ${fmt(tisRange[0], 5)} and ${fmt(tisRange[1], 5)} through every encounter from 1958 to 2010: it never moved faster than about
  ${fmt(vinf(tisRange[0]), 1)} km/s relative to the Earth. Luna 1's stage has ${fmt(D.luna1_T, 4)}, about ${fmt(vinf(D.luna1_T), 1)} km/s. To turn one into the other the Earth would have
  to shed most of that speed in a flyby, which a flyby cannot do. <b>Luna 1 is ruled out.</b> The same argument removes every stage that went straight from the Earth to
  interplanetary space (Mars, Venus and Pioneer launches).`;

  $("cloudText1").innerHTML = `Because the 1975 pass was slow and bound, it magnifies tiny differences enormously: orbits that agree to a fraction of an arcsecond in 2010
  disagree completely about 1974. No single back-tracked orbit can name a launch. So ${N} clones were drawn from the 2010 fit and each was run back to Sputnik's launch
  (1957 October 4) with all the planets, recording every pass by the Earth and the Moon. The question is then statistical: <i>what fraction of the orbits consistent with
  the data could have started on the Earth, and when?</i> A clone that comes from solar orbit and never approaches the Earth after 1957 cannot be a post-Sputnik
  launch; if 2010 KQ is artificial, as its spectrum and sunlight push say, it is not that clone.`;
  $("cloudTiles").innerHTML = tileHtml([["Clones run", `${N}`],
    ["Arrive from solar orbit, no Earth origin", `${nSolar} <small>(${pct(nSolar / N)})</small>`],
    ["Touch the Earth after 1957", `${nTouch} <small>(${pct(nTouch / N)}; ${pct(SUP.any.weighted)} weighted to Gray's July orbit)</small>`],
    ["Touch within 60 days after Luna 22", `${in22}`], ["Touch within 60 days after Luna 23", `${in23}`],
    ["Touch near Luna 1 or Luna 4", `${SUP.luna1.n + SUP.luna4.n}`]]);
  $("cloudText2").innerHTML = `"Touching" here means the backward run reaches the ground, or passes within 10,000 km of the Earth's centre (a real stage's departure burn starts in
  a parking orbit about 190 km up). The touching clones are almost all in the more strongly pushed half of the cloud (<i>C<sub>r</sub>A/m</i> above about 0.008), the half that
  matches Gray's July solution. Their touch dates run from late 1973 to the end of 1974 and bunch up on 1974 June 12&ndash;23 and October 29 &ndash; November 17, a few weeks
  after Luna 22 and Luna 23 reached the Moon. The bunching is partly built in &mdash; perigees of a looping orbit recur every 70&ndash;80 days, so many clones share them &mdash; and
  the exact dates should not be over-read: a real stage reaches its loose Earth orbit by a lunar flyby, which a slightly wrong clone replaces with a dive to the ground at its
  next perigee. What the cloud does say firmly is negative: <b>not one clone touches the Earth near Luna 4's launch in April 1963, nor stays bound to the Earth from 1963
  to 1975</b>, as a Luna 4 stage that never left would have to.`;

  $("susText1").innerHTML = `The cloud starts from 2010 KQ and runs backwards. The other direction starts from each suspect: put its stage on the lunar transfer it actually flew,
  give it a spread of miss distances at the Moon (the stage separates before any course correction and makes none), and fly it forward to the end of 1975 in the full model.
  The transfers start in the real parking orbits: planes set by the launch time, the launch site and the inclination (51.6&deg; for the Proton launches, 65&deg; for Luna 4's
  Molniya rocket), with the departure burn placed where a nearly horizontal burn reaches the Moon on the recorded arrival date. As a check, the Moon at arrival lies within
  ${fmt(Math.max(...Object.values(D.suspect_refs).map((r) => r.moon_plane_offset_deg)), 1)}&deg; of each parking-orbit plane, as it must for a correctly timed launch, and the Luna 4
  variants that pass the Moon at 7,000&ndash;10,000 km (Luna 4 itself missed by about 8,400 km) end up in Earth orbits of ${fmt(Math.min(...D.luna4_check.map((x) => x.r_min_km)) / 1e3)}&ndash;${fmt(Math.max(...D.luna4_check.map((x) => x.r_min_km)) / 1e3)}
  thousand by ${fmt(Math.min(...D.luna4_check.map((x) => x.r_max_km)) / 1e3)}&ndash;${fmt(Math.max(...D.luna4_check.map((x) => x.r_max_km)) / 1e3)} thousand km, bracketing the
  89,250 &times; 694,000 km actually recorded.`;
  $("susRefTable").innerHTML = table(["Suspect", "Departure burn (model)", "Speed", "Burn off horizontal", "Moon off plane", "Arrival (model)", "Variants"],
    SUSKEYS.map((k) => { const r = D.suspect_refs[k]; return [META[k].label, r.tli, `${fmt(r.v_kms, 3)} km/s`, `${fmt(r.fpa_deg, 1)}&deg;`, `${fmt(r.moon_plane_offset_deg, 1)}&deg;`, r.arrive, ST[k].n]; }),
    [2, 3, 4, 6]);
  $("susText2").innerHTML = table(["Fate by end of 1975", ...SUSKEYS.map((k) => META[k].label)], [
    ["hit the Earth or the Moon", ...SUSKEYS.map((k) => `${ST[k].imp} <small>(${pct(ST[k].imp / ST[k].n)})</small>`)],
    ["escaped before 1975", ...SUSKEYS.map((k) => `${ST[k].earlyEsc} <small>(${pct(ST[k].earlyEsc / ST[k].n)})</small>`)],
    ["still bound to the Earth on 1975 April 1", ...SUSKEYS.map((k) => `${ST[k].bound75} <small>(${pct(ST[k].bound75 / ST[k].n)})</small>`)],
    ["left the Earth in Apr 10 &ndash; Jun 30 1975, as 2010 KQ did", ...SUSKEYS.map((k) => `<b>${ST[k].esc75}</b> <small>(${pct(ST[k].esc75 / ST[k].n)})</small>`)],
    ["...onto 2010 KQ's 1976 orbit (&Delta;a, &Delta;e &lt; 0.005; &Delta;i &lt; 0.1&deg;)", ...SUSKEYS.map((k) => `${ST[k].match}`)],
    ["first perigee after the Moon (10&ndash;90%)", ...SUSKEYS.map((k) => ST[k].fp0 ? `${ST[k].fp0} &ndash; ${ST[k].fp1}` : "&ndash;")]], [1, 2, 3]);

  /*__VERDICT__*/

  $("futureText").innerHTML = `Run forward, all ${FW.length} clones return in the first days of January 2036, when 2010 KQ will again be about a million km away and moving slowly; every one
  is captured into Earth orbit again, as in 1975. ${FWimp} of them (${pct(FWimp / FW.length)}) then hit the Earth, almost all on 2036 April 6&ndash;10; weighting the clones
  towards Gray's July 2010 solution raises this to ${pct(fwW)}. Do not read this as a forecast. The share depends steeply on the sunlight push (almost none below 0.0078,
  about half to two thirds at 0.008&ndash;0.0095), which a tumbling, fading stage does not keep constant for 26 years, and on observations from 2010 that are not public. In
  2010 JPL's Paul Chodas put the chance of an impact in the 30 years from 2036 at 6%. The useful conclusion is practical: 2010 KQ will be back within reach of survey
  telescopes in late 2035, and a few weeks of fresh astrometry would pin down both its fate and, by tightening the 1975 pass, its origin.`;

  $("limitsList").innerHTML = [
    "Only 5 weeks of the 7-week arc are public. Gray's later solution (A/m 0.0085 &plusmn; 0.0015, closest approach 1975 April 27 at 152,000 km) is used only to weight clones, as a check.",
    "Sunlight pressure is a constant cannonball term. A tumbling stage presents a changing area, paint darkens over decades, and reflected and re-emitted heat (Yarkovsky) add small pushes; none is modelled. Earth and Moon shadows are ignored (short, near perigee).",
    "The clone cloud is a statistical statement about orbits consistent with the 2010 data under this model. It cannot follow any single orbit through the 1974&ndash;75 capture; its dates are indicative.",
    "The suspect transfers are reconstructions: parking orbits at 190 km and the standard inclinations, arrival times from the published mission dates, uniform miss distances up to 20,000 km. The real stages' separation and any avoidance burns are unknown.",
    "Physical size cuts the other way, weakly. A Blok-D (3.7 m &times; ~6 m) has an average cross-section near 23 m&sup2;, against 5&ndash;9 m&sup2; estimated from 2010 KQ's brightness by Miles, and a Blok-D's <i>C<sub>r</sub>A/m</i> should be ~0.01&ndash;0.016 against the measured 0.008. Luna 4's smaller Blok-L fits these better. Both estimates depend on reflectivity and attitude by factors of 2&ndash;3.",
    "Launches were checked against public lists for 1957&ndash;1975. A classified or failed launch that left a large stage in a high orbit in 1974 would not be in them."
  ].map((x) => `<li>${x}</li>`).join("");
})();
